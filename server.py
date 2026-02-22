#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
DeepSeek API - Provides an unofficial API for DeepSeek by reverse-engineering its web interface.
Copyright (C) 2025 smkttl

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
'''

import json
import time
import uuid
import os
from flask import Flask, request, jsonify, render_template, Response, stream_with_context
from DeepSeekAPI import DeepSeekChat

app = Flask(__name__)

# In-memory storage for persistent chat sessions (keyed by a client-supplied session id)
_chat_sessions: dict = {}


def _parse_credentials(auth_header: str):
    """Parse ds_session_id and authorization_token from an Authorization header.

    Expected format:  Bearer <ds_session_id>:<authorization_token>
    Returns (ds_session_id, authorization_token) or raises ValueError.
    """
    if not auth_header or not auth_header.startswith("Bearer "):
        raise ValueError(
            "Missing or invalid Authorization header. "
            "Expected: Bearer <ds_session_id>:<authorization_token>"
        )
    token_data = auth_header[7:]
    if ":" not in token_data:
        raise ValueError(
            "Invalid Authorization header. "
            "Expected format: Bearer <ds_session_id>:<authorization_token>"
        )
    ds_session_id, authorization_token = token_data.split(":", 1)
    return ds_session_id.strip(), authorization_token.strip()


def _get_session(session_key: str, ds_session_id: str, authorization_token: str) -> DeepSeekChat:
    """Return a cached DeepSeekChat session or create a new one."""
    if session_key not in _chat_sessions:
        _chat_sessions[session_key] = DeepSeekChat(ds_session_id, authorization_token)
    return _chat_sessions[session_key]


def _build_prompt(messages: list) -> str:
    """Convert an OpenAI messages list to a single prompt string."""
    if len(messages) == 1 and messages[0].get("role") == "user":
        return messages[0].get("content", "")
    parts = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            parts.append(f"System: {content}")
        elif role == "user":
            parts.append(f"User: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
    return "\n".join(parts)


def _openai_response(response_text: str, model: str, response_id: str = None):
    """Format a complete OpenAI chat completion response."""
    rid = response_id or f"chatcmpl-{uuid.uuid4().hex}"
    return {
        "id": rid,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": response_text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _sse_chunk(content: str, model: str, response_id: str, finish_reason=None):
    """Format a single SSE chunk in OpenAI streaming format."""
    delta = {} if finish_reason == "stop" else {"content": content}
    if content and finish_reason is None:
        delta["role"] = "assistant"
    chunk = {
        "id": response_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
    }
    return f"data: {json.dumps(chunk)}\n\n"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/v1/models", methods=["GET"])
def list_models():
    return jsonify(
        {
            "object": "list",
            "data": [
                {
                    "id": "deepseek-chat",
                    "object": "model",
                    "created": 1677610602,
                    "owned_by": "deepseek",
                },
                {
                    "id": "deepseek-reasoner",
                    "object": "model",
                    "created": 1677610602,
                    "owned_by": "deepseek",
                },
            ],
        }
    )


@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    # --- Parse request -------------------------------------------------
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": {"message": "Invalid JSON body", "type": "invalid_request_error"}}), 400

    messages = data.get("messages", [])
    model = data.get("model", "deepseek-chat")
    stream = data.get("stream", False)
    search_enabled = bool(data.get("search_enabled", False))
    thinking_enabled = model == "deepseek-reasoner"
    # Clients may also request a persistent session via a custom header/field
    session_key = data.get("session_id") or request.headers.get("X-Session-Id") or str(uuid.uuid4())

    # --- Parse credentials from Authorization header -------------------
    try:
        ds_session_id, authorization_token = _parse_credentials(
            request.headers.get("Authorization", "")
        )
    except ValueError as exc:
        return jsonify({"error": {"message": str(exc), "type": "auth_error"}}), 401

    if not messages:
        return jsonify({"error": {"message": "No messages provided", "type": "invalid_request_error"}}), 400

    prompt = _build_prompt(messages)
    if not prompt:
        return jsonify({"error": {"message": "Empty prompt", "type": "invalid_request_error"}}), 400

    # --- Build / reuse chat session ------------------------------------
    # Use a composite key so different credential pairs don't share sessions
    full_key = f"{ds_session_id}:{session_key}"
    chat = _get_session(full_key, ds_session_id, authorization_token)

    # --- Execute request -----------------------------------------------
    if stream:
        def generate():
            result = chat.send_message(prompt, False, thinking_enabled, search_enabled)
            rid = f"chatcmpl-{uuid.uuid4().hex}"
            if result and result.get("ok"):
                content = result["content"]
                thought = content.get("thought", "")
                response_text = content.get("response", "")
                if thought:
                    yield _sse_chunk(f"<think>\n{thought}\n</think>\n\n", model, rid)
                yield _sse_chunk(response_text, model, rid)
                yield _sse_chunk("", model, rid, finish_reason="stop")
                yield "data: [DONE]\n\n"
            else:
                error_msg = result.get("content", "Request failed") if result else "Request failed"
                yield f"data: {json.dumps({'error': str(error_msg)})}\n\n"

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    else:
        result = chat.send_message(prompt, False, thinking_enabled, search_enabled)
        if result and result.get("ok"):
            content = result["content"]
            thought = content.get("thought", "")
            response_text = content.get("response", "")
            full_response = (
                f"<think>\n{thought}\n</think>\n\n{response_text}" if thought else response_text
            )
            return jsonify(_openai_response(full_response, model))
        else:
            error_msg = result.get("content", "Request failed") if result else "Request failed"
            return jsonify({"error": {"message": str(error_msg), "type": "api_error"}}), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    debug = os.environ.get("DEBUG", "false").lower() == "true"
    print(f"Starting DeepSeek OpenAI-compatible API server on {host}:{port}")
    app.run(host=host, port=port, debug=debug)
