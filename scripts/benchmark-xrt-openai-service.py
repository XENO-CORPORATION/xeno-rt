#!/usr/bin/env python3
"""Exercise XRT's live OpenAI-compatible text service and retain evidence."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import http.client
import json
import math
import os
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


THINKING_PREFIXES = ("Thinking Process", "Here's a thinking process", "<think>")
REQUEST_MODEL = "Qwen3.6-27B"


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def process_rss_bytes(pid: int | None) -> int | None:
    if not pid:
        return None
    try:
        for line in Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        return None
    return None


def gpu_memory_bytes() -> int | None:
    try:
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
        ).strip().splitlines()[0]
        return int(raw) * 1024 * 1024
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


class ServiceClient:
    def __init__(self, base_url: str, timeout: float) -> None:
        parsed = urllib.parse.urlparse(base_url)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
            raise ValueError("service benchmark accepts only a loopback http URL")
        self.base_url = base_url.rstrip("/")
        self.host = parsed.hostname or "127.0.0.1"
        self.port = parsed.port or 80
        self.timeout = timeout

    def get_json(self, path: str) -> tuple[int, dict[str, Any], float]:
        start = time.perf_counter()
        with urllib.request.urlopen(self.base_url + path, timeout=self.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
            return response.status, payload, time.perf_counter() - start

    def post_json(
        self, path: str, payload: dict[str, Any]
    ) -> tuple[int, dict[str, Any] | str, float]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + path,
            data=body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8")
                return response.status, json.loads(raw), time.perf_counter() - start
        except urllib.error.HTTPError as error:
            raw = error.read().decode("utf-8", errors="replace")
            try:
                parsed: dict[str, Any] | str = json.loads(raw)
            except json.JSONDecodeError:
                parsed = raw
            return error.code, parsed, time.perf_counter() - start

    def stream_chat(
        self, payload: dict[str, Any], abort_after_content_chunks: int | None = None
    ) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        connection = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
        start = time.perf_counter()
        connection.request(
            "POST",
            "/v1/chat/completions",
            body=body,
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        chunks: list[str] = []
        reasoning_chunks: list[str] = []
        finish_reason = None
        first_content_seconds = None
        done = False
        aborted = False
        content_chunks = 0
        try:
            if response.status != 200:
                return {
                    "status": response.status,
                    "body": response.read().decode("utf-8", errors="replace"),
                    "elapsed_seconds": time.perf_counter() - start,
                }
            while True:
                line = response.readline()
                if not line:
                    break
                decoded = line.decode("utf-8", errors="replace").strip()
                if not decoded.startswith("data:"):
                    continue
                data = decoded[5:].strip()
                if data == "[DONE]":
                    done = True
                    break
                event = json.loads(data)
                choice = event.get("choices", [{}])[0]
                delta = choice.get("delta", {})
                content = delta.get("content")
                reasoning_content = delta.get("reasoning_content")
                if reasoning_content:
                    reasoning_chunks.append(reasoning_content)
                if content:
                    if first_content_seconds is None:
                        first_content_seconds = time.perf_counter() - start
                    chunks.append(content)
                    content_chunks += 1
                    if (
                        abort_after_content_chunks is not None
                        and content_chunks >= abort_after_content_chunks
                    ):
                        aborted = True
                        break
                if choice.get("finish_reason") is not None:
                    finish_reason = choice["finish_reason"]
        finally:
            response.close()
            connection.close()
        return {
            "status": response.status,
            "done": done,
            "aborted": aborted,
            "content": "".join(chunks),
            "reasoning_content": "".join(reasoning_chunks),
            "content_chunks": content_chunks,
            "ttft_seconds": first_content_seconds,
            "elapsed_seconds": time.perf_counter() - start,
            "finish_reason": finish_reason,
        }

    def stream_completion(self, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        connection = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
        start = time.perf_counter()
        connection.request(
            "POST",
            "/v1/completions",
            body=body,
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        chunks: list[str] = []
        finish_reason = None
        first_content_seconds = None
        done = False
        try:
            if response.status != 200:
                return {
                    "status": response.status,
                    "body": response.read().decode("utf-8", errors="replace"),
                    "elapsed_seconds": time.perf_counter() - start,
                }
            while True:
                line = response.readline()
                if not line:
                    break
                decoded = line.decode("utf-8", errors="replace").strip()
                if not decoded.startswith("data:"):
                    continue
                data = decoded[5:].strip()
                if data == "[DONE]":
                    done = True
                    break
                event = json.loads(data)
                choice = event.get("choices", [{}])[0]
                piece = choice.get("text")
                if piece:
                    if first_content_seconds is None:
                        first_content_seconds = time.perf_counter() - start
                    chunks.append(piece)
                if choice.get("finish_reason") is not None:
                    finish_reason = choice["finish_reason"]
        finally:
            response.close()
            connection.close()
        return {
            "status": response.status,
            "done": done,
            "content": "".join(chunks),
            "ttft_seconds": first_content_seconds,
            "elapsed_seconds": time.perf_counter() - start,
            "finish_reason": finish_reason,
        }


def chat_payload(prompt: str, max_tokens: int = 32) -> dict[str, Any]:
    return {
        "model": REQUEST_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "top_k": 1,
        "top_p": 1,
        "repetition_penalty": 1,
        "presence_penalty": 0,
        "frequency_penalty": 0,
        "seed": 424242,
        "enable_thinking": False,
    }


def main() -> int:
    global REQUEST_MODEL
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18080")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--server-pid", type=int)
    parser.add_argument("--model-path")
    parser.add_argument("--backend", default="cuda-resident")
    parser.add_argument("--request-model", default=REQUEST_MODEL)
    parser.add_argument("--long-context-suite", type=Path)
    parser.add_argument("--long-context-expected", type=Path)
    parser.add_argument(
        "--long-context-case",
        action="append",
        default=[],
        help="Run only this case ID from the long-context suite; repeat for multiple cases",
    )
    parser.add_argument("--soak-requests", type=int, default=30)
    parser.add_argument("--concurrency", default="1,2")
    parser.add_argument("--overload-concurrency", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--max-rss-growth-mib", type=int, default=256)
    parser.add_argument("--max-gpu-growth-mib", type=int, default=128)
    args = parser.parse_args()
    REQUEST_MODEL = args.request_model
    if bool(args.long_context_suite) != bool(args.long_context_expected):
        parser.error("--long-context-suite and --long-context-expected must be supplied together")
    client = ServiceClient(args.base_url, args.timeout)
    failures: list[dict[str, Any]] = []
    evidence: dict[str, Any] = {}
    started = time.time()

    def fail(gate: str, detail: Any) -> None:
        failures.append({"gate": gate, "detail": detail})

    try:
        _, models, models_seconds = client.get_json("/v1/models")
        _, initial_status, status_seconds = client.get_json("/v1/runtime/status")
        evidence["discovery"] = {
            "models": models,
            "models_seconds": models_seconds,
            "status": initial_status,
            "status_seconds": status_seconds,
        }
        if not initial_status.get("ready") or initial_status.get("active_backend") != "cuda-resident":
            fail("runtime_ready", initial_status)

        direct = chat_payload("Reply with exactly XRT_READY and nothing else.", 16)
        status, direct_body, elapsed = client.post_json("/v1/chat/completions", direct)
        evidence["non_streaming_direct"] = {
            "status": status,
            "elapsed_seconds": elapsed,
            "response": direct_body,
        }
        if status != 200 or not isinstance(direct_body, dict):
            fail("non_streaming_direct", direct_body)
        else:
            content = direct_body.get("choices", [{}])[0].get("message", {}).get("content", "")
            if "XRT_READY" not in content:
                fail("non_thinking_quality", content)
            if content.lstrip().startswith(THINKING_PREFIXES):
                fail("non_thinking_hygiene", content[:160])

        nested = chat_payload("Reply with exactly XRT_READY and nothing else.", 16)
        nested.pop("enable_thinking")
        nested["chat_template_kwargs"] = {"enable_thinking": False}
        nested_status, nested_body, nested_elapsed = client.post_json(
            "/v1/chat/completions", nested
        )
        evidence["non_streaming_nested"] = {
            "status": nested_status,
            "elapsed_seconds": nested_elapsed,
            "response": nested_body,
        }
        if status == 200 and nested_status == 200 and direct_body != nested_body:
            direct_content = direct_body.get("choices", [{}])[0].get("message", {}).get("content")
            nested_content = nested_body.get("choices", [{}])[0].get("message", {}).get("content")
            if direct_content != nested_content:
                fail("thinking_control_form_parity", [direct_content, nested_content])

        thinking_payload = chat_payload(
            "A service receives 240 requests per minute, rises by 35 percent, "
            "then caching removes 20 percent. How many backend requests remain?",
            1024,
        )
        thinking_payload["enable_thinking"] = True
        for field in (
            "temperature",
            "top_k",
            "top_p",
            "repetition_penalty",
            "presence_penalty",
            "frequency_penalty",
        ):
            thinking_payload.pop(field)
        thinking_status, thinking_body, thinking_elapsed = client.post_json(
            "/v1/chat/completions", thinking_payload
        )
        evidence["thinking_quality"] = {
            "status": thinking_status,
            "elapsed_seconds": thinking_elapsed,
            "response": thinking_body,
        }
        if thinking_status != 200 or not isinstance(thinking_body, dict):
            fail("thinking_quality_request", thinking_body)
        else:
            thinking_choice = thinking_body.get("choices", [{}])[0]
            thinking_message = thinking_choice.get("message", {})
            thinking_content = thinking_message.get("content", "")
            reasoning_content = thinking_message.get("reasoning_content", "")
            if "259.2" not in thinking_content[-256:]:
                fail("thinking_quality_answer", thinking_content)
            if not reasoning_content:
                fail("thinking_reasoning_content", thinking_message)
            if "We need" in thinking_content or "</think>" in thinking_content:
                fail("thinking_content_hygiene", thinking_content)
            if thinking_choice.get("finish_reason") != "stop":
                fail("thinking_quality_finish_reason", thinking_choice)

        thinking_stream_payload = {
            "model": args.request_model,
            "messages": [
                {
                    "role": "user",
                    "content": "Reply with exactly XRT_READY and nothing else.",
                }
            ],
            "max_tokens": 128,
            "seed": 424242,
            "enable_thinking": True,
            "stream": True,
        }
        thinking_streamed = client.stream_chat(thinking_stream_payload)
        evidence["thinking_streaming"] = thinking_streamed
        if thinking_streamed.get("status") != 200 or not thinking_streamed.get("done"):
            fail("thinking_streaming_contract", thinking_streamed)
        if "XRT_READY" not in thinking_streamed.get("content", ""):
            fail("thinking_streaming_answer", thinking_streamed)
        if not thinking_streamed.get("reasoning_content"):
            fail("thinking_streaming_reasoning", thinking_streamed)
        if "</think>" in thinking_streamed.get("content", ""):
            fail("thinking_streaming_hygiene", thinking_streamed)

        completion_payload = {
            "model": args.request_model,
            "prompt": "Continue this sequence with at least one token: alpha, beta, gamma,",
            "max_tokens": 16,
            "temperature": 0,
            "top_k": 1,
            "top_p": 1,
            "repetition_penalty": 1,
            "presence_penalty": 0,
            "frequency_penalty": 0,
            "seed": 424242,
        }
        completion_status, completion_body, completion_elapsed = client.post_json(
            "/v1/completions", completion_payload
        )
        evidence["completion_non_streaming"] = {
            "status": completion_status,
            "elapsed_seconds": completion_elapsed,
            "response": completion_body,
        }
        if completion_status != 200 or not isinstance(completion_body, dict):
            fail("completion_non_streaming", completion_body)
        else:
            choice = completion_body.get("choices", [{}])[0]
            if not choice.get("text"):
                fail("completion_non_streaming_content", choice)
            if choice.get("finish_reason") not in {"stop", "length"}:
                fail("completion_non_streaming_finish_reason", choice)
            usage = completion_body.get("usage", {})
            prompt_usage = usage.get("prompt_tokens", 0)
            completion_usage = usage.get("completion_tokens", 0)
            if (
                completion_usage <= 0
                or usage.get("total_tokens") != prompt_usage + completion_usage
            ):
                fail("completion_non_streaming_usage", usage)

        completion_stream_payload = dict(completion_payload)
        completion_stream_payload["max_tokens"] = 8
        completion_stream_payload["stream"] = True
        completion_streamed = client.stream_completion(completion_stream_payload)
        evidence["completion_streaming"] = completion_streamed
        if completion_streamed.get("status") != 200 or not completion_streamed.get("done"):
            fail("completion_streaming_contract", completion_streamed)
        if not completion_streamed.get("content"):
            fail("completion_streaming_content", completion_streamed)
        if completion_streamed.get("finish_reason") != "length":
            fail(
                "completion_streaming_length_finish_reason",
                completion_streamed.get("finish_reason"),
            )

        stream_payload = chat_payload(
            "Write the integers 1 through 100 in order, separated by commas.", 64
        )
        stream_payload["stream"] = True
        streamed = client.stream_chat(stream_payload)
        evidence["streaming"] = streamed
        if streamed.get("status") != 200 or not streamed.get("done"):
            fail("streaming_contract", streamed)
        if not streamed.get("content"):
            fail("streaming_content", streamed)
        if streamed.get("finish_reason") != "length":
            fail("streaming_length_finish_reason", streamed.get("finish_reason"))

        shared = "\n".join(
            f"Record {index:04d}: owner={(('amber', 'birch', 'cobalt', 'delta')[index % 4])}."
            for index in range(32)
        )
        history = [
            {"role": "system", "content": "Answer with only the requested value."},
            {"role": "user", "content": shared},
            {"role": "assistant", "content": "Context loaded."},
            {"role": "user", "content": "The release channel is cobalt."},
            {"role": "assistant", "content": "Channel recorded."},
        ]
        multi_results = []
        for question, expected in (
            ("What is the release channel?", "cobalt"),
            ("Who owns Record 0003?", "delta"),
            ("Who owns Record 0002?", "cobalt"),
        ):
            payload = chat_payload("unused", 24)
            payload["messages"] = history + [{"role": "user", "content": question}]
            code, body, seconds = client.post_json("/v1/chat/completions", payload)
            multi_results.append(
                {
                    "question": question,
                    "expected": expected,
                    "status": code,
                    "seconds": seconds,
                    "body": body,
                }
            )
            if code != 200:
                fail("multi_turn_request", multi_results[-1])
            elif isinstance(body, dict):
                content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
                if expected.casefold() not in content.casefold():
                    fail("multi_turn_quality", {"question": question, "expected": expected, "content": content})
        _, after_multiturn, _ = client.get_json("/v1/runtime/status")
        evidence["multi_turn"] = {
            "requests": multi_results,
            "prefix_cache_before": initial_status.get("prefix_cache"),
            "prefix_cache_after": after_multiturn.get("prefix_cache"),
        }

        if args.long_context_suite and args.long_context_expected:
            context_suite = json.loads(
                args.long_context_suite.read_text(encoding="utf-8")
            )
            context_expected = json.loads(
                args.long_context_expected.read_text(encoding="utf-8")
            )
            expected_by_case = {
                row["case_id"]: row for row in context_expected.get("cases", [])
            }
            available_cases = {
                case["id"] for case in context_suite.get("cases", [])
            }
            selected_cases = set(args.long_context_case)
            unknown_cases = sorted(selected_cases - available_cases)
            if unknown_cases:
                fail("long_context_case_selection", {"unknown_cases": unknown_cases})
            cases_to_run = [
                case
                for case in context_suite.get("cases", [])
                if not selected_cases or case["id"] in selected_cases
            ]
            context_rows = []
            for case in cases_to_run:
                case_id = case["id"]
                payload = chat_payload("unused", int(case.get("max_tokens") or 32))
                payload["messages"] = case["messages"]
                code, body, seconds = client.post_json("/v1/chat/completions", payload)
                content = ""
                usage = None
                if isinstance(body, dict):
                    content = (
                        body.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                    )
                    usage = body.get("usage")
                required = expected_by_case.get(case_id, {}).get("required_text")
                context_rows.append(
                    {
                        "case_id": case_id,
                        "status": code,
                        "seconds": seconds,
                        "required_text": required,
                        "content": content,
                        "usage": usage,
                        "error_body": body if code != 200 else None,
                    }
                )
                if code != 200 or not required or required.casefold() not in content.casefold():
                    fail("long_context_api", context_rows[-1])
            _, context_status, _ = client.get_json("/v1/runtime/status")
            evidence["long_context"] = {
                "suite": str(args.long_context_suite),
                "expected": str(args.long_context_expected),
                "selected_cases": sorted(selected_cases),
                "rows": context_rows,
                "post_request_status": context_status,
            }
            context_scheduler = context_status.get("scheduler", {})
            if (
                context_scheduler.get("active_sequences") != 0
                or context_scheduler.get("kv_reserved_bytes") != 0
            ):
                fail("long_context_cleanup", context_scheduler)

        concurrency_results: dict[str, Any] = {}
        for level in [int(value) for value in args.concurrency.split(",") if value.strip()]:
            ready = concurrent.futures.ThreadPoolExecutor(max_workers=level)
            wall_start = time.perf_counter()
            futures = [
                ready.submit(
                    client.post_json,
                    "/v1/chat/completions",
                    chat_payload(f"Reply with exactly concurrent-{level}-{index}.", 32),
                )
                for index in range(level)
            ]
            rows = []
            for future in futures:
                try:
                    code, body, seconds = future.result(timeout=args.timeout)
                    rows.append({"status": code, "seconds": seconds, "body": body})
                except Exception as error:  # noqa: BLE001 - evidence harness
                    rows.append({"status": None, "error": repr(error)})
            ready.shutdown(wait=True)
            wall_seconds = time.perf_counter() - wall_start
            latencies = [row["seconds"] for row in rows if "seconds" in row]
            concurrency_results[str(level)] = {
                "wall_seconds": wall_seconds,
                "successful": sum(row.get("status") == 200 for row in rows),
                "p50_seconds": percentile(latencies, 0.50),
                "p95_seconds": percentile(latencies, 0.95),
                "rows": rows,
            }
            if concurrency_results[str(level)]["successful"] != level:
                fail(f"concurrency_{level}", concurrency_results[str(level)])
            for index, row in enumerate(rows):
                body = row.get("body")
                if isinstance(body, dict):
                    content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
                    expected = f"concurrent-{level}-{index}"
                    if expected not in content:
                        fail(
                            f"concurrency_{level}_quality",
                            {"index": index, "expected": expected, "content": content},
                        )
        evidence["concurrency"] = concurrency_results

        overload_barrier = threading.Barrier(args.overload_concurrency + 1)

        def overload_request(index: int) -> tuple[int, dict[str, Any] | str, float]:
            overload_barrier.wait()
            payload = chat_payload(
                f"Write a detailed numbered list with 200 distinct items. Request {index}.",
                256,
            )
            return client.post_json("/v1/chat/completions", payload)

        overload_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=args.overload_concurrency
        )
        overload_futures = [
            overload_pool.submit(overload_request, index)
            for index in range(args.overload_concurrency)
        ]
        overload_barrier.wait()
        overload_rows = []
        for future in overload_futures:
            try:
                code, body, seconds = future.result(timeout=args.timeout)
                overload_rows.append({"status": code, "seconds": seconds, "body": body})
            except Exception as error:  # noqa: BLE001 - evidence harness
                overload_rows.append({"status": None, "error": repr(error)})
        overload_pool.shutdown(wait=True)
        overload_successes = sum(row.get("status") == 200 for row in overload_rows)
        overload_rejections = sum(row.get("status") == 429 for row in overload_rows)
        evidence["queue_overload"] = {
            "concurrency": args.overload_concurrency,
            "successful": overload_successes,
            "rejected_429": overload_rejections,
            "rows": overload_rows,
        }
        if overload_successes < 1 or overload_rejections < 1:
            fail("queue_overload_backpressure", evidence["queue_overload"])
        unexpected_overload = [
            row for row in overload_rows if row.get("status") not in {200, 429}
        ]
        if unexpected_overload:
            fail("queue_overload_status", unexpected_overload)

        cancel_payload = chat_payload(
            "Write a very long technical explanation of local inference with many sections.", 1024
        )
        cancel_payload["stream"] = True
        cancelled = client.stream_chat(cancel_payload, abort_after_content_chunks=1)
        drain_deadline = time.monotonic() + 30
        drained_status = None
        while time.monotonic() < drain_deadline:
            _, drained_status, _ = client.get_json("/v1/runtime/status")
            scheduler = drained_status.get("scheduler", {})
            if (
                scheduler.get("active_sequences") == 0
                and scheduler.get("queued_sequences") == 0
                and scheduler.get("kv_reserved_bytes") == 0
            ):
                break
            time.sleep(0.1)
        evidence["cancellation"] = {
            "client": cancelled,
            "drained_status": drained_status,
        }
        if not cancelled.get("aborted"):
            fail("cancellation_abort", cancelled)
        scheduler = (drained_status or {}).get("scheduler", {})
        if scheduler.get("active_sequences") != 0 or scheduler.get("kv_reserved_bytes") != 0:
            fail("cancellation_cleanup", scheduler)

        bad_status, bad_body, _ = client.post_json("/v1/chat/completions", {"max_tokens": 1})
        evidence["invalid_request"] = {"status": bad_status, "body": bad_body}
        if bad_status != 400:
            fail("invalid_request_status", evidence["invalid_request"])

        rss_before = process_rss_bytes(args.server_pid)
        gpu_before = gpu_memory_bytes()
        soak_rows = []
        output_hashes = []
        for index in range(args.soak_requests):
            code, body, seconds = client.post_json(
                "/v1/chat/completions",
                chat_payload("Reply with exactly SOAK_OK and nothing else.", 16),
            )
            content = ""
            if isinstance(body, dict):
                content = body.get("choices", [{}])[0].get("message", {}).get("content", "")
            digest = hashlib.sha256(content.encode("utf-8")).hexdigest()
            output_hashes.append(digest)
            soak_rows.append({"index": index, "status": code, "seconds": seconds, "digest": digest})
            if code != 200 or "SOAK_OK" not in content:
                fail("soak_request", {"index": index, "status": code, "content": content})
        rss_after = process_rss_bytes(args.server_pid)
        gpu_after = gpu_memory_bytes()
        _, final_status, _ = client.get_json("/v1/runtime/status")
        soak_latencies = [row["seconds"] for row in soak_rows]
        evidence["soak"] = {
            "requests": args.soak_requests,
            "successful": sum(row["status"] == 200 for row in soak_rows),
            "unique_output_hashes": sorted(set(output_hashes)),
            "mean_seconds": statistics.fmean(soak_latencies) if soak_latencies else None,
            "p50_seconds": percentile(soak_latencies, 0.50),
            "p95_seconds": percentile(soak_latencies, 0.95),
            "rss_before_bytes": rss_before,
            "rss_after_bytes": rss_after,
            "rss_delta_bytes": None if rss_before is None or rss_after is None else rss_after - rss_before,
            "gpu_before_bytes": gpu_before,
            "gpu_after_bytes": gpu_after,
            "gpu_delta_bytes": None if gpu_before is None or gpu_after is None else gpu_after - gpu_before,
            "rows": soak_rows,
        }
        rss_limit = args.max_rss_growth_mib * 1024 * 1024
        gpu_limit = args.max_gpu_growth_mib * 1024 * 1024
        if rss_before is not None and rss_after is not None and rss_after - rss_before > rss_limit:
            fail(
                "soak_rss_growth",
                {"delta_bytes": rss_after - rss_before, "limit_bytes": rss_limit},
            )
        if gpu_before is not None and gpu_after is not None and gpu_after - gpu_before > gpu_limit:
            fail(
                "soak_gpu_growth",
                {"delta_bytes": gpu_after - gpu_before, "limit_bytes": gpu_limit},
            )
        evidence["final_status"] = final_status
        final_scheduler = final_status.get("scheduler", {})
        if final_scheduler.get("active_sequences") != 0 or final_scheduler.get("kv_reserved_bytes") != 0:
            fail("final_scheduler_cleanup", final_scheduler)

        if args.model_path:
            gpu_before_unload = gpu_memory_bytes()
            unload_status, unload_body, unload_seconds = client.post_json(
                "/v1/runtime/unload", {}
            )
            _, unloaded_runtime, _ = client.get_json("/v1/runtime/status")
            unavailable_status, unavailable_body, _ = client.post_json(
                "/v1/chat/completions", chat_payload("Reply with OK.", 4)
            )
            gpu_after_unload = gpu_memory_bytes()
            load_status, load_body, load_seconds = client.post_json(
                "/v1/runtime/load",
                {"model_path": args.model_path, "backend": args.backend},
            )
            _, reloaded_runtime, _ = client.get_json("/v1/runtime/status")
            reload_probe_status, reload_probe_body, reload_probe_seconds = client.post_json(
                "/v1/chat/completions",
                chat_payload("Reply with exactly RELOAD_OK and nothing else.", 16),
            )
            _, post_reload_status, _ = client.get_json("/v1/runtime/status")
            evidence["lifecycle"] = {
                "unload": {
                    "status": unload_status,
                    "seconds": unload_seconds,
                    "body": unload_body,
                    "runtime_status": unloaded_runtime,
                    "gpu_before_bytes": gpu_before_unload,
                    "gpu_after_bytes": gpu_after_unload,
                },
                "unavailable_probe": {
                    "status": unavailable_status,
                    "body": unavailable_body,
                },
                "reload": {
                    "status": load_status,
                    "seconds": load_seconds,
                    "body": load_body,
                    "runtime_status": reloaded_runtime,
                    "post_probe_status": post_reload_status,
                },
                "reload_probe": {
                    "status": reload_probe_status,
                    "seconds": reload_probe_seconds,
                    "body": reload_probe_body,
                },
            }
            if unload_status != 200 or unloaded_runtime.get("ready"):
                fail("lifecycle_unload", evidence["lifecycle"]["unload"])
            if unavailable_status != 503:
                fail("lifecycle_unavailable_status", evidence["lifecycle"]["unavailable_probe"])
            if load_status != 200 or not reloaded_runtime.get("ready"):
                fail("lifecycle_reload", evidence["lifecycle"]["reload"])
            if reload_probe_status != 200 or not isinstance(reload_probe_body, dict):
                fail("lifecycle_reload_probe", evidence["lifecycle"]["reload_probe"])
            else:
                reload_content = (
                    reload_probe_body.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                if "RELOAD_OK" not in reload_content:
                    fail("lifecycle_reload_quality", reload_content)
            post_reload_scheduler = post_reload_status.get("scheduler", {})
            if (
                post_reload_scheduler.get("active_sequences") != 0
                or post_reload_scheduler.get("kv_reserved_bytes") != 0
            ):
                fail("lifecycle_reload_cleanup", post_reload_scheduler)
    except Exception as error:  # noqa: BLE001 - evidence must survive failures
        fail("harness_exception", repr(error))

    report = {
        "object": "xrt.openai_service_admission",
        "schema_version": 1,
        "started_unix": started,
        "elapsed_seconds": time.time() - started,
        "base_url": args.base_url,
        "server_pid": args.server_pid,
        "passed": not failures,
        "failures": failures,
        "evidence": evidence,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({"passed": report["passed"], "failures": failures}, ensure_ascii=False))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
