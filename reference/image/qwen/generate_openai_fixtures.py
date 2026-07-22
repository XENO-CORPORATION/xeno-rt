#!/usr/bin/env python3
"""Generate deterministic OpenAI image wire fixtures with the pinned SDK."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import tempfile
from email import policy
from email.parser import BytesParser
from pathlib import Path
from typing import Any, Callable

import httpx
import openai
from openai import OpenAI
from PIL import Image


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[2]
LOCK_PATH = HERE / "phase0-lock.json"
OUTPUT_DIR = REPO_ROOT / "tests" / "fixtures" / "openai" / "images"
AVATAR_1 = REPO_ROOT / "tests" / "fixtures" / "image-quality" / "avatar-01.png"
AVATAR_2 = REPO_ROOT / "tests" / "fixtures" / "image-quality" / "avatar-02.png"
MASK_1 = REPO_ROOT / "tests" / "fixtures" / "image-quality" / "mask-01.png"


def canonical_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode()


def atomic_write(path: Path, body: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


class Capture:
    def __init__(self, responder: Callable[[httpx.Request], httpx.Response]) -> None:
        self.requests: list[httpx.Request] = []
        self.responder = responder

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.responder(request)


def client_for(capture: Capture) -> OpenAI:
    return OpenAI(
        api_key="fixture-api-key",
        base_url="https://fixture.invalid/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(capture)),
        max_retries=0,
    )


def request_json(request: httpx.Request) -> dict[str, Any]:
    return {
        "method": request.method,
        "path": request.url.path,
        "content_type": request.headers["content-type"],
        "body": json.loads(request.content),
    }


def normalized_multipart(request: httpx.Request) -> dict[str, Any]:
    content_type = request.headers["content-type"]
    message = BytesParser(policy=policy.default).parsebytes(
        b"Content-Type: " + content_type.encode("ascii") + b"\r\nMIME-Version: 1.0\r\n\r\n" + request.content
    )
    parts = []
    for part in message.iter_parts():
        name = part.get_param("name", header="content-disposition")
        filename = part.get_filename()
        payload = part.get_payload(decode=True) or b""
        record: dict[str, Any] = {
            "name": name,
            "filename": filename,
            "content_type": part.get_content_type(),
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        if filename is None:
            record["value"] = payload.decode("utf-8")
        parts.append(record)
    return {
        "method": request.method,
        "path": request.url.path,
        "content_type": "multipart/form-data; boundary=<sdk-generated>",
        "part_order": [part["name"] for part in parts],
        "parts": parts,
    }


def sse(events: list[dict[str, Any]]) -> bytes:
    text = "".join(
        "event: "
        + event["type"]
        + "\n"
        + "data: "
        + json.dumps(event, sort_keys=True, separators=(",", ":"))
        + "\n\n"
        for event in events
    )
    return text.encode()


def usage(image_tokens: int) -> dict[str, Any]:
    return {
        "input_tokens": 19 + image_tokens,
        "input_tokens_details": {"image_tokens": image_tokens, "text_tokens": 19},
        "output_tokens": 1024,
        "total_tokens": 1043 + image_tokens,
    }


def generation_fixtures(image_b64: str, second_image_b64: str, webp_b64: str) -> dict[str, bytes]:
    response = {
        "created": 1784670000,
        "background": "opaque",
        "data": [{"b64_json": image_b64}, {"b64_json": second_image_b64}],
        "output_format": "png",
        "quality": "high",
        "size": "1024x1024",
        "usage": usage(0),
    }
    capture = Capture(lambda request: httpx.Response(200, json=response, request=request))
    client = client_for(capture)
    parsed = client.images.generate(
        model="qwen-image-2512-q4_k_m",
        prompt="A cobalt mechanical keyboard on a walnut desk.",
        background="opaque",
        n=2,
        output_format="png",
        quality="high",
        response_format="b64_json",
        size="1024x1024",
        user="fixture-user",
        extra_body={"x_xeno": {"backend": "cuda", "seed": 424242, "steps": 50}},
    )
    if parsed.model_dump(exclude_none=True) != response:
        raise RuntimeError("pinned SDK changed the generation response shape")
    request = request_json(capture.requests[0])
    client.close()

    compressed_response = {
        "created": 1784670003,
        "background": "opaque",
        "data": [{"b64_json": webp_b64}],
        "output_format": "webp",
        "quality": "medium",
        "size": "1024x1024",
        "usage": usage(0),
    }
    compressed_capture = Capture(
        lambda captured_request: httpx.Response(
            200, json=compressed_response, request=captured_request
        )
    )
    compressed_client = client_for(compressed_capture)
    compressed_parsed = compressed_client.images.generate(
        model="qwen-image-2512-q4_k_m",
        prompt="A compact satellite unfolding above Earth.",
        background="opaque",
        n=1,
        output_compression=80,
        output_format="webp",
        quality="medium",
        size="1024x1024",
    )
    if compressed_parsed.model_dump(exclude_none=True) != compressed_response:
        raise RuntimeError("pinned SDK changed the compressed generation response shape")
    compressed_request = request_json(compressed_capture.requests[0])
    compressed_client.close()

    stream_events = [
        {
            "type": "image_generation.partial_image",
            "b64_json": image_b64,
            "background": "opaque",
            "created_at": 1784670001,
            "output_format": "png",
            "partial_image_index": 0,
            "quality": "high",
            "size": "1024x1024",
        },
        {
            "type": "image_generation.completed",
            "b64_json": image_b64,
            "background": "opaque",
            "created_at": 1784670002,
            "output_format": "png",
            "quality": "high",
            "size": "1024x1024",
            "usage": usage(0),
        },
    ]
    stream_capture = Capture(
        lambda request: httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=sse(stream_events),
            request=request,
        )
    )
    stream_client = client_for(stream_capture)
    stream = stream_client.images.generate(
        model="qwen-image-2512-q4_k_m",
        prompt="A cobalt mechanical keyboard on a walnut desk.",
        background="opaque",
        output_format="png",
        partial_images=1,
        quality="high",
        size="1024x1024",
        stream=True,
    )
    observed_events = [event.model_dump(exclude_none=True) for event in stream]
    stream.close()
    stream_client.close()
    if observed_events != stream_events:
        raise RuntimeError("pinned SDK changed the generation SSE event shape")

    return {
        "generation-request.json": canonical_bytes(request),
        "generation-response.json": canonical_bytes(response),
        "generation-compressed-request.json": canonical_bytes(compressed_request),
        "generation-compressed-response.json": canonical_bytes(compressed_response),
        "generation-stream-request.json": canonical_bytes(request_json(stream_capture.requests[0])),
        "generation-stream-events.json": canonical_bytes(stream_events),
        "generation-stream.sse": sse(stream_events),
    }


def edit_fixtures(image_b64: str, second_image_b64: str, mask_b64: str) -> dict[str, bytes]:
    response = {
        "created": 1784670100,
        "background": "opaque",
        "data": [{"b64_json": image_b64}],
        "output_format": "png",
        "quality": "high",
        "size": "1024x1024",
        "usage": usage(2048),
    }
    capture = Capture(lambda request: httpx.Response(200, json=response, request=request))
    client = client_for(capture)
    parsed = client.images.edit(
        image=[
            ("avatar-01.png", AVATAR_1.read_bytes(), "image/png"),
            ("avatar-02.png", AVATAR_2.read_bytes(), "image/png"),
        ],
        mask=("mask-01.png", MASK_1.read_bytes(), "image/png"),
        prompt="Place both people in a bright studio and preserve their ordering.",
        model="qwen-image-edit-2511-q4_k_m",
        background="opaque",
        input_fidelity="high",
        n=1,
        output_compression=100,
        output_format="png",
        quality="high",
        response_format="b64_json",
        size="1024x1024",
        user="fixture-user",
        extra_body={"x_xeno": {"backend": "cuda", "seed": 515151, "steps": 50}},
    )
    if parsed.model_dump(exclude_none=True) != response:
        raise RuntimeError("pinned SDK changed the edit response shape")
    request = normalized_multipart(capture.requests[0])
    client.close()

    stream_events = [
        {
            "type": "image_edit.partial_image",
            "b64_json": image_b64,
            "background": "opaque",
            "created_at": 1784670101,
            "output_format": "png",
            "partial_image_index": 0,
            "quality": "high",
            "size": "1024x1024",
        },
        {
            "type": "image_edit.completed",
            "b64_json": image_b64,
            "background": "opaque",
            "created_at": 1784670102,
            "output_format": "png",
            "quality": "high",
            "size": "1024x1024",
            "usage": usage(2048),
        },
    ]
    stream_capture = Capture(
        lambda request: httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=sse(stream_events),
            request=request,
        )
    )
    stream_client = client_for(stream_capture)
    stream = stream_client.images.edit(
        image=("avatar-01.png", AVATAR_1.read_bytes(), "image/png"),
        prompt="Replace only the background with a modern library.",
        model="qwen-image-edit-2511-q4_k_m",
        background="opaque",
        output_format="png",
        partial_images=1,
        quality="high",
        size="1024x1024",
        stream=True,
    )
    observed_events = [event.model_dump(exclude_none=True) for event in stream]
    stream.close()
    stream_client.close()
    if observed_events != stream_events:
        raise RuntimeError("pinned SDK changed the edit SSE event shape")

    json_request = {
        "method": "POST",
        "path": "/v1/images/edits",
        "content_type": "application/json",
        "body": {
            "images": [
                {"image_url": "data:image/png;base64," + image_b64},
                {"image_url": "data:image/png;base64," + second_image_b64},
            ],
            "mask": {"image_url": "data:image/png;base64," + mask_b64},
            "prompt": "Place both people in a bright studio and preserve their ordering.",
            "model": "qwen-image-edit-2511-q4_k_m",
            "background": "opaque",
            "input_fidelity": "high",
            "moderation": "auto",
            "n": 1,
            "output_compression": 100,
            "output_format": "png",
            "quality": "high",
            "size": "1024x1024",
            "user": "fixture-user",
            "x_xeno": {"backend": "cuda", "seed": 515151, "steps": 50},
        },
    }

    return {
        "edit-request-json.json": canonical_bytes(json_request),
        "edit-request-multipart.json": canonical_bytes(request),
        "edit-response.json": canonical_bytes(response),
        "edit-stream-request-multipart.json": canonical_bytes(
            normalized_multipart(stream_capture.requests[0])
        ),
        "edit-stream-events.json": canonical_bytes(stream_events),
        "edit-stream.sse": sse(stream_events),
    }


def error_fixture() -> dict[str, bytes]:
    response = {
        "error": {
            "message": "The style parameter is not supported by this local Qwen image profile.",
            "type": "invalid_request_error",
            "param": "style",
            "code": "unsupported_parameter",
        }
    }
    capture = Capture(lambda request: httpx.Response(400, json=response, request=request))
    client = client_for(capture)
    try:
        client.images.generate(
            model="qwen-image-2512-q4_k_m",
            prompt="fixture",
            style="vivid",
        )
    except openai.BadRequestError as error:
        observed = {
            "status_code": error.status_code,
            "code": error.code,
            "param": error.param,
            "type": error.type,
        }
    else:
        raise RuntimeError("pinned SDK did not raise BadRequestError for the fixture")
    finally:
        client.close()
    expected = {
        "status_code": 400,
        "code": "unsupported_parameter",
        "param": "style",
        "type": "invalid_request_error",
    }
    if observed != expected:
        raise RuntimeError(f"pinned SDK changed error extraction: {observed!r}")
    return {
        "unsupported-parameter-error.json": canonical_bytes(response),
        "unsupported-parameter-sdk-observation.json": canonical_bytes(expected),
    }


def build_outputs() -> dict[str, bytes]:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    expected_version = lock["openai_clients"]["python"]
    if openai.__version__ != expected_version:
        raise RuntimeError(f"expected openai=={expected_version}, observed {openai.__version__}")
    image_b64 = base64.b64encode(AVATAR_1.read_bytes()).decode("ascii")
    second_image_b64 = base64.b64encode(AVATAR_2.read_bytes()).decode("ascii")
    mask_b64 = base64.b64encode(MASK_1.read_bytes()).decode("ascii")
    webp_buffer = io.BytesIO()
    with Image.open(AVATAR_1) as source:
        source.convert("RGB").save(webp_buffer, format="WEBP", quality=80, method=6)
    webp_b64 = base64.b64encode(webp_buffer.getvalue()).decode("ascii")
    outputs = {}
    outputs.update(generation_fixtures(image_b64, second_image_b64, webp_b64))
    outputs.update(edit_fixtures(image_b64, second_image_b64, mask_b64))
    outputs.update(error_fixture())
    manifest = {
        "schema_version": 2,
        "status": "frozen",
        "openai_python": expected_version,
        "openai_python_commit": lock["git_pins"]["openai-python"]["commit"],
        "openai_node": lock["openai_clients"]["node"],
        "openai_node_commit": lock["git_pins"]["openai-node"]["commit"],
        "generation_transport": "application/json",
        "edit_transports": ["application/json", "multipart/form-data"],
        "server_openapi": {
            "version": "2.3.0",
            "observed_at": "2026-07-22",
            "generation_endpoint": "https://api.openai.com/v1/images/generations",
            "edit_endpoint": "https://api.openai.com/v1/images/edits",
        },
        "files": {
            name: {"size_bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}
            for name, body in sorted(outputs.items())
        },
    }
    outputs["fixture-manifest.json"] = canonical_bytes(manifest)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if not args.verify and not args.write:
        parser.error("select --write and/or --verify")
    outputs = build_outputs()
    if args.write:
        for name, body in outputs.items():
            atomic_write(OUTPUT_DIR / name, body)
    if args.verify:
        for name, expected in outputs.items():
            try:
                observed = (OUTPUT_DIR / name).read_bytes()
            except FileNotFoundError as error:
                raise RuntimeError(f"missing OpenAI fixture: {name}") from error
            if observed != expected:
                raise RuntimeError(f"OpenAI fixture drift: {name}")
    print(
        json.dumps(
            {
                "status": "ok",
                "mode": "write+verify" if args.write and args.verify else "write" if args.write else "verify",
                "fixtures": len(outputs),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
