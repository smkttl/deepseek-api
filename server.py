#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
DeepSeek API Server - OpenAI-compatible server for DeepSeek
'''

import os
import json
from flask import Flask, request, jsonify, Response, stream_with_context
from DeepSeekAPI import DeepSeekChat, DeepSeekChatIOMethods

app = Flask(__name__)

# Load tokens from file or environment
def get_tokens():
    if os.path.exists("tokens"):
        with open("tokens") as f:
            lines = f.read().strip().split('\n')
            if len(lines) >= 2:
                return lines[0], lines[1]
    ds_session_id = os.environ.get("DS_SESSION_ID")
    authorization_token = os.environ.get("AUTHORIZATION_TOKEN")
    if ds_session_id and authorization_token:
        return ds_session_id, authorization_token
    raise ValueError("Tokens not found. Set DS_SESSION_ID and AUTHORIZATION_TOKEN or create 'tokens' file.")

DS_SESSION_ID, AUTHORIZATION_TOKEN = get_tokens()

def build_messages(messages):
    """Convert OpenAI-style messages to DeepSeek format"""
    result = []
    for msg in messages:
        if msg["role"] == "system":
            result.append({"role": "system", "content": msg["content"]})
        elif msg["role"] == "user":
            result.append({"role": "user", "content": msg["content"]})
        elif msg["role"] == "assistant":
            result.append({"role": "assistant", "content": msg["content"]})
    return result

def chat_non_streaming(messages, thinking_enabled=True):
    """Non-streaming chat"""
    user_message = messages[-1]["content"] if messages else ""
    chat = DeepSeekChat(DS_SESSION_ID, AUTHORIZATION_TOKEN)
    result = chat.send_message(user_message, printing=False, thinking_enabled=thinking_enabled, search_enabled=False)
    return result

def chat_streaming(messages, thinking_enabled=True):
    """Streaming chat using SSE"""
    user_message = messages[-1]["content"] if messages else ""
    chat = DeepSeekChat(DS_SESSION_ID, AUTHORIZATION_TOKEN)
    
    def generate():
        import subprocess
        import sys
        import io
        
        # Use subprocess to capture stdout
        proc = subprocess.Popen(
            [
                sys.executable, '-c',
                f"from DeepSeekAPI import DeepSeekChat; "
                f"DeepSeekChat({repr(DS_SESSION_ID)}, {repr(AUTHORIZATION_TOKEN)}).send_message("
                f"{repr(user_message)}, True, {str(thinking_enabled).lower()}, False)"
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=False
        )
        
        buffer = ""
        while True:
            char = proc.stdout.read(1)
            if not char:
                break
            char = char.decode('utf-8', errors='replace')
            buffer += char
            
            # Parse complete messages from buffer
            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                if line.startswith('data:'):
                    data = line[5:].strip()
                    if data:
                        yield f"data: {json.dumps({'choices': [{'delta': {'content': data}}]})}\n\n"
        
        proc.wait()
        yield "data: [DONE]\n\n"
    
    return generate()

@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    data = request.json
    
    messages = data.get("messages", [])
    stream = data.get("stream", False)
    thinking_enabled = not data.get("disable_thinking", False)
    
    # Handle model field (for compatibility)
    model = data.get("model", "deepseek-chat")
    
    if stream:
        return Response(
            stream_with_context(chat_streaming(messages, thinking_enabled)),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
            }
        )
    else:
        result = chat_non_streaming(messages, thinking_enabled)
        
        return jsonify({
            "id": f"chatcmpl-{hash(str(messages))}",
            "object": "chat.completion",
            "created": int(__import__("time").time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result or ""
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        })

@app.route("/v1/models", methods=["GET"])
def list_models():
    return jsonify({
        "object": "list",
        "data": [{
            "id": "deepseek-chat",
            "object": "model",
            "created": 1704067200,
            "owned_by": "deepseek"
        }]
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="DeepSeek API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    args = parser.parse_args()
    
    print(f"Starting DeepSeek API Server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port)
