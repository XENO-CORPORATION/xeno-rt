"""Narrate a script through the durable job API. Python 3 standard library only.

    python narrate_job.py reference.wav script.txt narration.wav [--base URL]

Safe to re-run: the Idempotency-Key is derived from the inputs, so re-running
after a crash or network error resumes waiting on the SAME job instead of
starting a second render. Existing output files are never overwritten.
Exit codes: 0 success, 2 job failed/cancelled/interrupted, 3 server not ready.
"""
import argparse
import base64
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("script", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--base", default="http://127.0.0.1:3338")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--timeout-minutes", type=float, default=120)
    a = parser.parse_args()
    voice_id = "narrator-" + hashlib.sha256(a.reference.read_bytes()).hexdigest()[:32]
    for path in (a.output, a.output.with_suffix(".json")):
        if path.exists():
            parser.error(f"refusing to overwrite {path}")

    def call(method, path, body=None, headers=None, timeout=120):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(a.base + path, data=data, method=method,
                                     headers={**({} if data is None else {"Content-Type": "application/json"}), **(headers or {})})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return response.status, json.load(response)
        except urllib.error.HTTPError as error:
            try:
                return error.code, json.load(error)
            except ValueError:
                return error.code, {"error": {"message": error.reason}}

    code, status = call("GET", "/v1/audio/status")
    if code != 200:
        print("server not ready:", code, status, file=sys.stderr)
        return 3
    print("runtime:", json.dumps(status))

    code, voice = call("GET", "/v1/audio/voices/" + voice_id)
    if code == 404:
        code, voice = call("POST", "/v1/audio/voices", {
            "id": voice_id, "name": "Documentary narrator",
            "audio_b64": base64.b64encode(a.reference.read_bytes()).decode("ascii")})
    if code not in (200, 201):
        print("voice registration failed:", code, voice, file=sys.stderr)
        return 3
    print("voice:", voice["id"], voice["sha256"])

    script = a.script.read_text(encoding="utf-8-sig")
    request = {"model": "chatterbox-multilingual-v3", "input": script, "voice": voice_id,
               "language": "en", "device": a.device, "preset": "documentary", "seed": 42,
               "word_check": True, "best_effort": False, "timeout_seconds": 7200}
    key = "narrate-" + hashlib.sha256(json.dumps(request, sort_keys=True).encode()
                                      + voice["sha256"].encode()).hexdigest()[:48]
    code, job = call("POST", "/v1/audio/jobs", request, {"Idempotency-Key": key})
    if code not in (200, 202):
        print("submit failed:", code, job, file=sys.stderr)
        return 2
    print("job:", job["id"], "(resumed)" if code == 200 else "(new)")

    deadline = time.monotonic() + a.timeout_minutes * 60
    last = None
    while True:
        code, job = call("GET", "/v1/audio/jobs/" + job["id"])
        if code != 200:
            print("status failed:", code, job, file=sys.stderr)
            return 2
        if (job["status"], job.get("completed_chunks")) != last:
            last = (job["status"], job.get("completed_chunks"))
            print("status:", job["status"], "chunks done:", job.get("completed_chunks", 0))
        if job["status"] == "succeeded":
            break
        if job["status"] in ("failed", "cancelled", "interrupted"):
            print("job ended:", job["status"], job.get("error"), file=sys.stderr)
            return 2
        if time.monotonic() > deadline:
            call("POST", f"/v1/audio/jobs/{job['id']}/cancel", {})
            print("client deadline reached; cancellation requested", file=sys.stderr)
            return 2
        time.sleep(2)

    code, result = call("GET", job["result_url"], timeout=600)
    if code != 200:
        print("result failed:", code, result, file=sys.stderr)
        return 2
    audio = base64.b64decode(result.pop("audio_b64"), validate=True)
    for path, data in ((a.output, audio), (a.output.with_suffix(".json"), json.dumps(result, indent=2).encode())):
        tmp = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
        with tmp.open("xb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        # Atomic create-only publication also refuses a concurrent writer.
        os.link(tmp, path)
        tmp.unlink()
    problems = [c["word_problems"] for c in result.get("chunks", []) if c.get("word_problems")]
    print("saved", a.output, f"{result['duration']:.2f}s", "provider", result["provider"],
          "word differences to review:" if problems else "no word differences", problems or "")
    # Terminal records are retained until deleted; remove it once saved.
    call("DELETE", "/v1/audio/jobs/" + job["id"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
