"""Live audio HTTP contract test, using only Python's standard library.

Run against a dedicated xrt-server test process with a separate voice root:
  python scripts/test-audio-api.py --base http://127.0.0.1:3711 \
      --reference reference.wav --output evidence-directory
Requires actual Chatterbox v3 + Whisper models. Missing models fail, never skip.
Creates a unique test voice, exercises it, and deletes only that voice in finally.
No external service is contacted by this script; the base must be loopback.
"""
import argparse
import base64
import concurrent.futures
import json
import math
import os
from pathlib import Path
import struct
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

TEXT = (
    "A balance beam tilts as the attendant steadies it. The heart rests on one side, "
    "and a feather rests on the other. This is not a confession.\r\n\r\n"
    "The speaker describes a life of truth and justice. Listen to the claim, "
    "and then consider what the scales are meant to decide."
)


def save_new(path, data):
    if path.exists():
        raise RuntimeError(f"refusing to overwrite evidence: {path}")
    tmp = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    with tmp.open("xb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:3711")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    url = urllib.parse.urlsplit(args.base)
    if url.scheme != "http" or url.hostname not in ("127.0.0.1", "localhost", "::1") or url.username:
        parser.error("use a dedicated loopback HTTP server")
    args.output.mkdir(parents=True, exist_ok=False)
    checks = []
    voice_id = "api-test-" + uuid.uuid4().hex
    clip = base64.b64encode(args.reference.read_bytes()).decode("ascii")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def call(method, path, body=None, timeout=1800):
        data = None if body is None else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(args.base.rstrip("/") + path, data=data, method=method,
                                         headers={} if data is None else {"Content-Type": "application/json"})
        try:
            with opener.open(request, timeout=timeout) as response:
                return response.status, dict(response.headers), response.read()
        except urllib.error.HTTPError as error:
            return error.code, dict(error.headers), error.read()

    def check(name, condition):
        checks.append({"name": name, "passed": bool(condition)})
        print(("PASS " if condition else "FAIL ") + name, flush=True)
        if not condition:
            raise AssertionError(name)

    created = False
    started = time.monotonic()
    try:
        code, _, raw = call("POST", "/v1/audio/voices", {"id": voice_id, "audio_b64": clip})
        check("register real reference", code == 201)
        created = True
        manifest = json.loads(raw)
        check("persistent voice identity and checksum", manifest["id"] == voice_id and len(manifest["sha256"]) == 64)
        code, _, _ = call("POST", "/v1/audio/voices", {"id": voice_id, "audio_b64": clip})
        check("no voice overwrite", code == 409)
        code, _, raw = call("GET", "/v1/audio/voices/" + voice_id)
        check("read registered identity", code == 200 and json.loads(raw) == manifest)
        code, _, raw = call("GET", "/v1/audio/voices")
        check("list registered identity", code == 200 and voice_id in [v["id"] for v in json.loads(raw)["data"]])
        for name, fields, expected in [
            ("empty script", {"input": ""}, 400),
            ("unknown model", {"model": "not-chatterbox"}, 400),
            ("wrong Chatterbox version", {"model": "chatterbox-v2"}, 400),
            ("unsupported speed", {"speed": 0.9}, 400),
            ("shaping without alignment", {"word_check": False, "breath_reduction_db": 12}, 400),
            ("unknown saved voice", {"voice": voice_id + "-missing"}, 404),
            ("unsafe saved voice", {"voice": "../escape"}, 400),
        ]:
            request = {"input": TEXT, "voice": voice_id, "device": args.device, **fields}
            code, _, _ = call("POST", "/v1/audio/speech", request)
            check(name + " refused", code == expected)

        request = {"input": TEXT, "voice": voice_id, "device": args.device,
                   "preset": "dramatic", "response_format": "json", "seed": 42}
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            synthesis = pool.submit(call, "POST", "/v1/audio/speech", request)
            # Probe until the inference worker owns admission. Probes use an
            # invalid script, so cannot launch a second expensive generation.
            busy = False
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline and not synthesis.done():
                code, headers, _ = call("POST", "/v1/audio/speech", {"input": ""}, timeout=10)
                if code == 429:
                    busy = any(k.lower() == "retry-after" for k in headers)
                    break
                time.sleep(0.05)
            check("concurrent request receives bounded busy response", busy)
            code, headers, raw = synthesis.result()
        if code != 200:
            print(raw[:2000].decode("utf-8", errors="replace"), flush=True)
        check("real generation succeeds", code == 200)
        result = json.loads(raw)
        audio = base64.b64decode(result.pop("audio_b64"), validate=True)
        check("WAV returned", audio[:4] == b"RIFF" and audio[8:12] == b"WAVE")
        check("explicit device honored", result["provider"].startswith(args.device) and result["asr_provider"].startswith(args.device))
        check("all chunks checked without best effort", bool(result["chunks"]) and all(
            c["word_error_rate"] is not None and not c["best_effort"] for c in result["chunks"]))
        words = result["words"]
        check("caption text preserves script", " ".join(w["text"] for w in words) == " ".join(TEXT.split()))
        check("timestamps finite and within output", bool(words) and all(
            math.isfinite(w["start"]) and math.isfinite(w["end"]) and
            0 <= w["start"] <= w["end"] <= result["duration"] + 0.001 for w in words))
        check("word starts monotonic", all(a["start"] <= b["start"] for a, b in zip(words, words[1:])))
        check("word check header present", any(k.lower() == "x-xrt-word-check" for k in headers))
        save_new(args.output / "speech.wav", audio)
        save_new(args.output / "speech.json", json.dumps(result, indent=2).encode("utf-8"))

        code, _, pcm = call("POST", "/v1/audio/speech", {
            "input": "The evidence remains inside the speaker's own chest.",
            "voice_b64": clip, "device": args.device, "response_format": "pcm", "seed": 7})
        check("inline reference and PCM response", code == 200 and len(pcm) > 24000 and len(pcm) % 2 == 0)
        samples = struct.unpack("<" + "h" * (len(pcm) // 2), pcm)
        check("PCM contains non-silent signed samples", min(samples) < -100 and max(samples) > 100)
        save_new(args.output / "speech.pcm", pcm)
    finally:
        if created:
            code, _, _ = call("DELETE", "/v1/audio/voices/" + voice_id)
            check("delete test voice", code == 200)
            code, _, _ = call("GET", "/v1/audio/voices/" + voice_id)
            check("deleted voice absent", code == 404)
        save_new(args.output / "checks.json", json.dumps({"elapsed_seconds": time.monotonic() - started,
                   "checks": checks}, indent=2).encode("utf-8"))
    print(f"{len(checks)}/{len(checks)} passed", flush=True)


if __name__ == "__main__":
    main()
