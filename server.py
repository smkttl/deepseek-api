#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
DeepSeek API Server - OpenAI-compatible server for DeepSeek
'''

import os
import json
import time
import sys
from io import StringIO
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

def get_thinking_enabled(model: str) -> bool:
    """Determine if thinking is enabled based on model name"""
    model_lower = model.lower()
    # deepseek-r1 enables thinking, deepseek-v3 disables it
    if 'r1' in model_lower or 'reasoning' in model_lower:
        return True
    return False

def chat_non_streaming(messages, thinking_enabled=True):
    """Non-streaming chat"""
    user_message = messages[-1]["content"] if messages else ""
    
    # Redirect stdout to capture output
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    
    try:
        chat = DeepSeekChat(DS_SESSION_ID, AUTHORIZATION_TOKEN)
        chat.send_message(user_message, printing=True, thinking_enabled=thinking_enabled, search_enabled=False)
    finally:
        sys.stdout = old_stdout
    
    output = mystdout.getvalue()
    
    # Parse output to extract response
    in_response = False
    response_text = ""
    
    for line in output.split('\n'):
        if 'START RESPONSE' in line:
            in_response = True
            continue
        elif 'FINISHED' in line:
            in_response = False
            break
        elif 'START THINK' in line:
            in_response = False
            continue
        
        if in_response and line.strip():
            response_text += line + '\n'
    
    return response_text.strip() if response_text else output

def chat_streaming(messages, thinking_enabled=True):
    """Streaming chat using SSE"""
    user_message = messages[-1]["content"] if messages else ""
    
    # Redirect stdout to capture output
    old_stdout = sys.stdout
    sys.stdout = mystdout = StringIO()
    
    try:
        chat = DeepSeekChat(DS_SESSION_ID, AUTHORIZATION_TOKEN)
        chat.send_message(user_message, printing=True, thinking_enabled=thinking_enabled, search_enabled=False)
    finally:
        sys.stdout = old_stdout
    
    output = mystdout.getvalue()
    
    # Parse and yield tokens
    in_thinking = False
    in_response = False
    
    for line in output.split('\n'):
        if 'START THINK' in line:
            in_thinking = True
            in_response = False
            continue
        elif 'START RESPONSE' in line:
            in_thinking = False
            in_response = True
            continue
        elif 'FINISHED' in line:
            in_response = False
            break
        
        if in_thinking:
            continue  # Skip thinking in streaming for now
        elif in_response:
            if line.strip():
                content_line = line + '\n'
                data_str = json.dumps({'choices': [{'delta': {'content': content_line}}]})
                yield "data: " + data_str + "\n\n"
    
    yield "data: [DONE]\n\n"

@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    data = request.json
    
    messages = data.get("messages", [])
    stream = data.get("stream", False)
    
    # Determine model - default to deepseek-chat
    model = data.get("model", "deepseek-chat")
    
    # Determine thinking enabled based on model name
    thinking_enabled = get_thinking_enabled(model)
    
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
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": len(result.split()) if result else 0,
                "total_tokens": len(result.split()) if result else 0
            }
        })

@app.route("/v1/models", methods=["GET"])
def list_models():
    return jsonify({
        "object": "list",
        "data": [
            {
                "id": "deepseek-v3",
                "object": "model",
                "created": 1704067200,
                "owned_by": "deepseek",
                "description": "DeepSeek V3 - Fast responses without extended thinking"
            },
            {
                "id": "deepseek-r1",
                "object": "model",
                "created": 1704067200,
                "owned_by": "deepseek",
                "description": "DeepSeek R1 - Reasoning model with extended thinking"
            }
        ]
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
