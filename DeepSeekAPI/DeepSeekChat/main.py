#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DeepSeek API - Unofficial client for chat.deepseek.com (reverse-engineered web SSE).

Debug-friendly rewrite:
- Every stage returns {"ok": bool, "stage": str, ...}
- Every parser decision is recorded in a _Trace
- Raw SSE lines are always preserved
- trace_dump() formats everything for bug reports

License: AGPL-3.0-or-later
"""

import utils.transport
import utils.session
import json
import os
import time
from dataclasses import dataclass, field
from traceback import format_exc as backtrace
from typing import Any, Callable, Dict, List, Optional

import requests


# ---------------------------------------------------------------------------
# Result helpers
# ---------------------------------------------------------------------------
def fail(stage: str, error: str, **extra) -> Dict[str, Any]:
    out = {"ok": False, "stage": stage, "error": error}
    out.update(extra)
    return out


def ok(stage: str, **extra) -> Dict[str, Any]:
    out = {"ok": True, "stage": stage}
    out.update(extra)
    return out


# ---------------------------------------------------------------------------
# Trace
# ---------------------------------------------------------------------------
@dataclass
class _Trace:
    """Records every parser decision for post-mortem debugging."""
    events: List[Dict[str, Any]] = field(default_factory=list)

    def add(self, kind: str, **fields):
        entry = {"t": round(time.time(), 3), "kind": kind}
        entry.update(fields)
        self.events.append(entry)

    def dump(self) -> str:
        lines = [f"# Trace ({len(self.events)} events)"]
        for i, e in enumerate(self.events):
            kind = e.pop("kind")
            t = e.pop("t")
            body = " ".join(f"{k}={v!r}" for k, v in e.items())
            lines.append(f"[{i:04d}] {t}  {kind:<10}  {body}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------
class DeepSeekChat:
    BASE_URL = "https://chat.deepseek.com"
    VALID_OPS = {"APPEND", "SET", "REPLACE", "BATCH", "DELETE", None}
    VALID_MODEL_TYPES = ("default", "expert")

    def __init__(self, ds_session_id: str, authorization_token: str,
                 debug: bool = False, timeout: int = 60):
        self.session = requests.Session()
        self.session.cookies.set("ds_session_id", ds_session_id)
        self.authorization = authorization_token
        self.debug = debug or bool(os.environ.get("DEEPSEEK_DEBUG"))
        self.timeout = timeout
        self.headers = {
            "accept": "*/*",
            "accept-language": "zh-CN,zh;q=0.9,fr;q=0.8",
            "authorization": self.authorization,
            "content-type": "application/json",
            "priority": "u=1, i",
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-origin",
            "x-app-version": "20241129.1",
            "x-client-locale": "zh_CN",
            "x-client-platform": "web",
            "x-client-version": "1.5.0",
            "x-debug-lite-model-channel": "prod",
            "x-debug-model-channel": "prod",
            "user-agent": (
                "Mozilla/5.0 (X11; Linux x86_64; rv:109.0) "
                "Gecko/20100101 Firefox/142.0"
            ),
        }
        self.chat_session_id: Optional[str] = None
        self.parent_message_id: Optional[str] = None
        # Last trace is kept here so callers can inspect it after send_message.
        self.last_trace: Optional[_Trace] = None

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------
    def _log(self, msg: str):
        if self.debug:
            print(f"[deepseek] {msg}", flush=True)

    def _post_json(self, url: str, payload: dict, extra_headers: dict = None,
                   stage: str = "post") -> Dict[str, Any]:
        headers = self.headers.copy()
        if extra_headers:
            headers.update(extra_headers)
        try:
            resp = self.session.post(
                url, headers=headers, data=json.dumps(payload),
                timeout=self.timeout,
            )
        except requests.RequestException as e:
            return fail(stage, f"network error: {e}")
        self._log(f"{stage}: HTTP {resp.status_code} ({len(resp.content)} bytes)")
        if resp.status_code != 200:
            return fail(stage, f"HTTP {resp.status_code}",
                        http={"status": resp.status_code, "body": resp.text[:2000]})
        try:
            body = resp.json()
        except ValueError:
            return fail(stage, "response was not JSON",
                        http={"status": 200, "body": resp.text[:2000]})
        return ok(stage, body=body, http_status=200)

    # ------------------------------------------------------------------
    # Stage 1: session
    # ------------------------------------------------------------------
    def create_chat_session(self) -> Dict[str, Any]:
        if self.chat_session_id:
            return ok("create_chat_session", chat_session_id=self.chat_session_id,
                      note="reused existing session")
        url = f"{self.BASE_URL}/api/v0/chat_session/create"
        r = self._post_json(url, {}, stage="create_chat_session")
        if not r["ok"]:
            return r
        body = r["body"]
        if body.get("code") != 0:
            return fail("create_chat_session", "non-zero code in body", body=body)
        try:
            self.chat_session_id = (
                body["data"]["biz_data"]["chat_session"]["id"]
            )
        except (KeyError, TypeError) as e:
            return fail("create_chat_session", f"unexpected body shape: {e}",
                        body=body)
        return ok("create_chat_session", chat_session_id=self.chat_session_id)

    # ------------------------------------------------------------------
    # Stage 2: PoW challenge
    # ------------------------------------------------------------------
    def create_pow_challenge(self) -> Dict[str, Any]:
        if not self.chat_session_id:
            return fail("create_pow_challenge", "no chat_session_id; call "
                        "create_chat_session() first")
        url = f"{self.BASE_URL}/api/v0/chat/create_pow_challenge"
        headers = {"referrer": f"{self.BASE_URL}/a/chat/s/{self.chat_session_id}"}
        r = self._post_json(url, {"target_path": "/api/v0/chat/completion"},
                            extra_headers=headers, stage="create_pow_challenge")
        if not r["ok"]:
            return r
        body = r["body"]
        if body.get("code") != 0:
            return fail("create_pow_challenge", "non-zero code in body", body=body)
        try:
            challenge = body["data"]["biz_data"]["challenge"]
        except (KeyError, TypeError) as e:
            return fail("create_pow_challenge", f"unexpected body shape: {e}",
                        body=body)
        return ok("create_pow_challenge", challenge=challenge)

    # ------------------------------------------------------------------
    # Stage 3: PoW solve
    # ------------------------------------------------------------------
    def solve_pow_challenge(self, challenge: dict) -> Dict[str, Any]:
        try:
            from .DeepSeekWASM import solve_wasm  # noqa: F401
        except ImportError as e:
            return fail("solve_pow_challenge",
                        f"missing DeepSeekWASM helper: {e}. "
                        "Place DeepSeekWASM.py next to this file.")
        required = ("algorithm", "challenge", "salt", "expire_at",
                    "difficulty", "signature", "target_path")
        missing = [k for k in required if k not in challenge]
        if missing:
            return fail("solve_pow_challenge",
                        f"challenge missing fields: {missing}", challenge=challenge)
        try:
            value, pow_response = solve_wasm(
                challenge["algorithm"], challenge["challenge"],
                challenge["salt"], challenge["expire_at"],
                challenge["difficulty"], challenge["signature"],
                challenge["target_path"],
            )
        except Exception as e:
            return fail("solve_pow_challenge", f"WASM solver raised: {e}",
                        tb=backtrace())
        if not value:
            return fail("solve_pow_challenge", "WASM solver returned falsy",
                        pow_response=pow_response)
        return ok("solve_pow_challenge", pow_response=pow_response)

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def send_message(self, message: str, printing: Optional[Callable] = None,
                     thinking_enabled: bool = False,
                     search_enabled: bool = False,
                     model_type: str = "default") -> Dict[str, Any]:
        """Send a message and return a fully-structured result dict.

        On success the dict contains:
            ok=True, stage='done', response=<str>, thought=<str>,
            citation=<dict>, title=<str>, thinktime=<float>,
            tokencount=<int|None>, trace=<str>, state=<dict>

        On failure the dict contains:
            ok=False, stage='<failed stage>', error=<str>, plus any
            stage-specific extras (http, body, challenge, tb, ...).
        """
        self.last_trace = _Trace()
        trace = self.last_trace

        if model_type not in self.VALID_MODEL_TYPES:
            return fail("validate", f"model_type must be one of "
                        f"{self.VALID_MODEL_TYPES}, got {model_type!r}")

        # --- session ---
        r = self.create_chat_session()
        trace.add("session", result=r)
        if not r["ok"]:
            return self._with_trace(r, trace)

        # --- challenge ---
        r = self.create_pow_challenge()
        trace.add("challenge", result=r)
        if not r["ok"]:
            return self._with_trace(r, trace)
        challenge = r["challenge"]

        # --- solve ---
        r = self.solve_pow_challenge(challenge)
        trace.add("solve", result=r)
        if not r["ok"]:
            return self._with_trace(r, trace)
        pow_response = r["pow_response"]

        # --- stream ---
        r = self._stream_completion(
            message=message, pow_response=pow_response,
            thinking_enabled=thinking_enabled,
            search_enabled=search_enabled,
            model_type=model_type,
            printing=printing, trace=trace,
        )
        if not r["ok"]:
            return self._with_trace(r, trace)

        state = r["state"]
        if state["msgid"]:
            self.parent_message_id = state["msgid"]
        else:
            trace.add("parent_message_id",
                      note="stream ended without message_id; keeping old parent")

        ordered = self._ordered_fragments(state)
        thought = "".join(f["content"] for f in ordered
                          if f.get("type") == "THINK" and f.get("content"))
        respond = "".join(f["content"] for f in ordered
                          if f.get("type") == "RESPONSE" and f.get("content"))

        result = ok(
            "done",
            response=respond,
            thought=thought,
            citation=state["citation"],
            title=state["title"],
            thinktime=state["thinktime"],
            tokencount=state["tokencount"],
            thinking_enabled=thinking_enabled,
            search_enabled=search_enabled,
            model_type=model_type,
            parent_message_id=self.parent_message_id,
            state=state,
        )
        return self._with_trace(result, trace)

    @staticmethod
    def _with_trace(result: Dict[str, Any], trace: _Trace) -> Dict[str, Any]:
        result["trace"] = trace.dump()
        return result

    # ------------------------------------------------------------------
    # Stage 4: SSE consumption
    # ------------------------------------------------------------------
    def _stream_completion(self, *, message: str, pow_response: str,
                           thinking_enabled: bool, search_enabled: bool,
                           model_type: str,
                           printing: Optional[Callable],
                           trace: _Trace) -> Dict[str, Any]:
        url = f"{self.BASE_URL}/api/v0/chat/completion"
        headers = self.headers.copy()
        headers["x-ds-pow-response"] = pow_response
        headers["accept"] = "text/event-stream"
        headers["referrer"] = f"{self.BASE_URL}/a/chat/s/{self.chat_session_id}"

        payload = {
            "chat_session_id": self.chat_session_id,
            "parent_message_id": self.parent_message_id,
            "model_type": model_type,
            "prompt": message,
            "ref_file_ids": [],
            "thinking_enabled": thinking_enabled,
            "search_enabled": search_enabled,
            "action": None,
            "preempt": False,
        }
        trace.add("completion_request", url=url, payload=payload)

        try:
            resp = self.session.post(url, headers=headers,
                                     data=json.dumps(payload),
                                     timeout=self.timeout, stream=True)
        except requests.RequestException as e:
            return fail("completion", f"network error: {e}")

        trace.add("completion_response",
                  status=resp.status_code,
                  content_type=resp.headers.get("content-type"))

        if resp.status_code != 200:
            return fail("completion", f"HTTP {resp.status_code}",
                        http={"status": resp.status_code,
                              "body": resp.text[:4000]})

        if "text/event-stream" not in resp.headers.get("content-type", ""):
            body = b"".join(resp.iter_content(chunk_size=8192)).decode(
                "utf-8", "replace")
            return fail("completion", "response was not text/event-stream",
                        http={"status": 200, "body": body[:4000]})

        state = self._new_state()
        event = "UNKNOWN"
        emit = (lambda s: print(s, end="", flush=True)) if printing else (lambda s: None)

        try:
            for lineno, line in enumerate(resp.iter_lines(decode_unicode=True)):
                if line is None:
                    continue
                state["raw_lines"].append(line)

                if not isinstance(line, str):
                    trace.add("bad_line", lineno=lineno, type=str(type(line)),
                              repr=repr(line))
                    return fail("stream", f"non-string line at {lineno}",
                                state=state)

                if not line:
                    continue

                if line.startswith("event: "):
                    event = line[7:].strip()
                    trace.add("event", name=event)
                    if event == "close":
                        trace.add("close", note="breaking loop on close event")
                        break
                    continue

                if line.startswith("data: "):
                    raw = line[6:].strip()
                    if not raw:
                        continue
                    try:
                        chunk = json.loads(raw)
                    except json.JSONDecodeError as e:
                        trace.add("bad_json", raw=raw, error=str(e))
                        return fail("stream", f"invalid JSON in SSE data: {e}",
                                    raw=raw, state=state)

                    outcome = self._handle_envelope(event, chunk, state, emit,
                                                    search_enabled, trace)
                    if outcome is not None and not outcome.get("ok", True):
                        return outcome

                    if event == "close":
                        break
                    continue

                if line.startswith(":"):
                    # SSE comment / keepalive
                    trace.add("keepalive", line=line)
                    continue

                trace.add("unknown_line", line=line)
                # Don't fail — unknown lines are recorded and skipped.

        except Exception as e:
            trace.add("stream_exception", error=str(e), tb=backtrace())
            return fail("stream", f"unhandled stream error: {e}",
                        tb=backtrace(), state=state)

        trace.add("stream_end", raw_line_count=len(state["raw_lines"]))
        return ok("stream", state=state)

    @staticmethod
    def _new_state() -> Dict[str, Any]:
        return {
            "msgid": None,
            "parid": None,
            "reqid": None,
            "resid": None,
            "title": "",
            "thinktime": 0,
            "tokencount": None,
            "citation": {},
            "fragments": {},        # index(str) -> {"type": str|None, "content": str}
            "active_fragment": None,
            "raw_lines": [],
        }

    # ------------------------------------------------------------------
    # Envelope dispatch
    # ------------------------------------------------------------------
    def _handle_envelope(self, event: str, chunk: Any, state: Dict[str, Any],
                         emit: Callable, search_enabled: bool,
                         trace: _Trace) -> Optional[Dict[str, Any]]:
        """Return None on success, or a fail() dict to abort the stream."""
        try:
            if event == "ready" and isinstance(chunk, dict):
                state["reqid"] = chunk.get("request_message_id", state["reqid"])
                state["resid"] = chunk.get("response_message_id", state["resid"])
                trace.add("ready", reqid=state["reqid"], resid=state["resid"])
                return None
            if event == "title" and isinstance(chunk, dict):
                state["title"] = chunk.get("content", state["title"])
                trace.add("title", title=state["title"])
                return None
            if event in ("finish", "close"):
                trace.add(event, chunk=chunk)
                return None
            # update_session, ready-with-payload, unknown events: parse
            self._parse(chunk, state, emit, search_enabled, trace)
            return None
        except Exception as e:
            trace.add("envelope_exception", event=event, chunk=chunk,
                      error=str(e), tb=backtrace())
            return fail("parse", f"error in event {event!r}: {e}",
                        tb=backtrace(), chunk=chunk, state=state)

    # ------------------------------------------------------------------
    # Recursive parser
    # ------------------------------------------------------------------
    def _parse(self, node: Any, state: Dict[str, Any], emit: Callable,
               search_enabled: bool, trace: _Trace, depth: int = 0):

               
        if depth > 50:
            trace.add("depth_guard", note="recursion depth exceeded")
            return
        if node is None or isinstance(node, bool):
            return
        if isinstance(node, list):
            for item in node:
                self._parse(item, state, emit, search_enabled, trace, depth + 1)
            return
        if isinstance(node, str):
            self._append_active(node, state, emit, trace)
            return
        if not isinstance(node, dict):
            trace.add("skip_non_dict", type=str(type(node)), repr=repr(node))
            return

        # --- metadata side-channels (never short-circuit) ---
        if "message_id" in node:
            state["msgid"] = node["message_id"]
        if "parent_id" in node:
            state["parid"] = node["parent_id"]

        # --- path-based dispatch ---
        if "p" in node:
            self._handle_path(node, state, emit, search_enabled, trace)
            return

        # --- type envelope: opens/selects a fragment ---
        if node.get("type"):
            self._handle_type_envelope(node, state, emit, search_enabled, trace)
            return

        # --- bare value ---
        if set(node.keys()) == {"v"}:
            self._parse(node["v"], state, emit, search_enabled, trace, depth + 1)
            return

        # --- wrapped response ---
        if "response" in node:
            self._parse(node["response"], state, emit, search_enabled,
                        trace, depth + 1)
            return

        # --- pure metadata ---
        if set(node.keys()) == {"updated_at"}:
            trace.add("metadata", updated_at=node["updated_at"])
            return

        trace.add("unhandled_dict", keys=list(node.keys()), node=node)

    def _append_active(self, text: str, state: Dict[str, Any],
                       emit: Callable, trace: _Trace):
        idx = state["active_fragment"]
        if idx is None:
            trace.add("preamble_token", value=text)
            return
        frag = state["fragments"].setdefault(idx, {"type": None, "content": ""})
        frag["content"] += text
        trace.add("append", fragment=idx, value=text, total_len=len(frag["content"]))
        emit(text)

    def _handle_type_envelope(self, node: dict, state: Dict[str, Any],
                              emit: Callable, search_enabled: bool,
                              trace: _Trace):
        frag_index = node.get("fragment_index")
        if frag_index is None:
            frag_index = str(len(state["fragments"]))
        frag_index = str(frag_index)
        frag = state["fragments"].setdefault(frag_index,
                                              {"type": None, "content": ""})
        frag["type"] = node["type"]
        state["active_fragment"] = frag_index
        trace.add("type_envelope", type=node["type"], fragment=frag_index,
                  had_content="content" in node)
        if self.debug:
            emit(f"\n--- START {node['type']} (fragment {frag_index}) ---\n")
        if "content" in node:
            self._parse(node["content"], state, emit, search_enabled, trace)

    # ------------------------------------------------------------------
    # Path handler
    # ------------------------------------------------------------------
    def _handle_path(self, node: dict, state: Dict[str, Any], emit: Callable,
                     search_enabled: bool, trace: _Trace):
        path = node["p"]
        op = node.get("o")
        value = node.get("v")

        if op not in self.VALID_OPS:
            trace.add("unknown_op", path=path, op=op, value=value)
            return

        parts = path.split("/")
        trace.add("path", path=path, op=op, value_repr=repr(value)[:200])

        # --- response/fragments/<index>/<field> ---
        if len(parts) >= 4 and parts[1] == "fragments":
            idx, field = str(parts[2]), parts[3]
            frag = state["fragments"].setdefault(idx,
                                                  {"type": None, "content": ""})
            if field == "content":
                old = frag["content"]
                frag["content"] = self._apply_op(old, op, value or "")
                state["active_fragment"] = idx
                trace.add("fragment_content", fragment=idx, op=op,
                          appended=repr(value)[:120], total_len=len(frag["content"]))
                if op in (None, "APPEND") and value:
                    emit(value)
            elif field == "type":
                frag["type"] = value
                state["active_fragment"] = idx
                trace.add("fragment_type", fragment=idx, type=value)
            elif field == "references" and search_enabled and isinstance(value, list):
                self._collect_citations(value, state, emit, trace)
            elif field == "status":
                trace.add("fragment_status", fragment=idx, status=value)
                if self.debug:
                    emit(f"\n--- fragment {idx} status: {value} ---\n")
            else:
                trace.add("fragment_field_ignored", fragment=idx, field=field,
                          value_repr=repr(value)[:120])
            return

        # --- response/<field> ---
        if len(parts) == 2:
            key = parts[1]
            if key == "status":
                trace.add("status", value=value)
                if self.debug:
                    emit(f"\n--- status: {value} ---\n")
            elif key == "accumulated_token_usage":
                state["tokencount"] = value
                trace.add("tokens", value=value)
            elif key == "elapsed_secs":
                state["thinktime"] = value
                trace.add("thinktime", value=value)
            elif key == "results" and search_enabled and isinstance(value, list):
                self._collect_citations(value, state, emit, trace)
            else:
                trace.add("field_ignored", key=key, value_repr=repr(value)[:120])
            return

        # --- any/other/path ---
        trace.add("path_unhandled", path=path, op=op,
                  value_repr=repr(value)[:200])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _apply_op(existing: Any, op: Optional[str], value: Any) -> Any:
        if op in (None, "APPEND"):
            if isinstance(existing, str) and isinstance(value, str):
                return existing + value
            if isinstance(existing, list) and isinstance(value, list):
                return existing + value
            return value if value is not None else existing
        if op in ("SET", "REPLACE"):
            return value
        if op == "DELETE":
            return "" if isinstance(existing, str) else None
        return value

    @staticmethod
    def _fragment_sort_key(index: str):
        try:
            return (0, int(index))
        except (TypeError, ValueError):
            return (1, str(index))

    def _ordered_fragments(self, state: Dict[str, Any]) -> List[dict]:
        return [
            state["fragments"][k]
            for k in sorted(state["fragments"], key=self._fragment_sort_key)
        ]

    def _collect_citations(self, items: list, state: Dict[str, Any],
                           emit: Callable, trace: _Trace):
        for item in items:
            if not isinstance(item, dict) or "url" not in item:
                trace.add("citation_skip", item=item)
                continue
            idx = item.get("cite_index")
            if idx is None:
                idx = len(state["citation"])
            state["citation"][idx] = item
            title = item.get("title", item.get("site_name", "UNKNOWN"))
            site = item.get("site_name", "UNKNOWN")
            trace.add("citation", index=idx, url=item["url"], title=title)
            emit(f"{idx}. [{title} - {site}]({item['url']})\n")
            snippet = item.get("snippet", "")
            if snippet:
                emit(f"> {snippet}\n")


# ---------------------------------------------------------------------------
# CLI entry point for manual testing
# ---------------------------------------------------------------------------
def _main():
    import argparse

    ap = argparse.ArgumentParser(description="DeepSeekChat debug client")
    ap.add_argument("--session-id", required=True, help="ds_session_id cookie")
    ap.add_argument("--auth", required=True, help="Bearer authorization token")
    ap.add_argument("--prompt", required=True, help="user message")
    ap.add_argument("--think", action="store_true", help="enable thinking")
    ap.add_argument("--search", action="store_true", help="enable search")
    ap.add_argument("--model", default="default", choices=("default", "expert"))
    ap.add_argument("--no-print", action="store_true", help="don't stream to stdout")
    ap.add_argument("--debug", action="store_true", help="verbose logging")
    args = ap.parse_args()

    chat = DeepSeekChat(args.session_id, args.auth, debug=args.debug)
    printing = None if args.no_print else (lambda s: print(s, end="", flush=True))

    result = chat.send_message(
        args.prompt, printing=printing,
        thinking_enabled=args.think, search_enabled=args.search,
        model_type=args.model,
    )

    print("\n\n===== RESULT =====")
    print(f"ok:     {result['ok']}")
    print(f"stage:  {result['stage']}")
    if result["ok"]:
        print(f"title:  {result.get('title')!r}")
        print(f"tokens: {result.get('tokencount')}")
        print(f"think:  {result.get('thinktime')}s")
        print(f"len(response): {len(result['response'])}")
        print(f"len(thought):  {len(result['thought'])}")
        print(f"citations:     {len(result.get('citation', {}))}")
    else:
        print(f"error:  {result['error']}")
        for k in ("http", "body", "challenge", "raw", "chunk", "tb"):
            if k in result:
                print(f"\n--- {k} ---")
                print(result[k])

    if args.debug or not result["ok"]:
        print("\n===== TRACE =====")
        print(result.get("trace", "(no trace)"))


if __name__ == "__main__":
    _main()