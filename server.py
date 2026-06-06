#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
DeepSeek API Server - OpenAI-compatible server for DeepSeek
'''

import os
import re
import json
import time
import sys
import queue
import threading
from io import StringIO
from flask import Flask, request, jsonify, Response, stream_with_context, render_template
from DeepSeekAPI import DeepSeekChat

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

def get_model_config(model: str):
    """Map OpenAI-compatible model names to DeepSeek web model flags."""
    model_lower = model.lower()
    model_type = "expert" if "v4" in model_lower or "r4" in model_lower or "expert" in model_lower else "default"
    thinking_enabled = "r1" in model_lower or "r4" in model_lower or "reasoning" in model_lower or "reasoner" in model_lower
    return model_type, thinking_enabled

class StreamingToolParser:
    def __init__(self):
        self.in_tool_calls = False
        self.normal_buffer = ""
        self.tool_buffer = ""
        self.yielded_name = False
        self.tool_call_id = f"call_{int(time.time())}"
        self.arg_start_idx = -1
        self.arg_yielded_len = 0
        self.brace_count = 0
        self.done = False

    def feed(self, chunk):
        if self.done:
            return chunk, None, None
            
        if not self.in_tool_calls:
            self.normal_buffer += chunk
            if "[tool_calls]" in self.normal_buffer:
                parts = self.normal_buffer.split("[tool_calls]", 1)
                normal_delta = parts[0]
                self.in_tool_calls = True
                self.normal_buffer = ""
                rest = parts[1]
                tool_res = self._feed_tool(rest)
                if isinstance(tool_res, tuple):
                    tool_delta, finish_reason, after_normal = tool_res
                    return normal_delta + after_normal, tool_delta, finish_reason
                else:
                    return normal_delta, tool_res, None
            else:
                tag = "[tool_calls]"
                hold_len = 0
                for i in range(1, len(tag)):
                    suffix = self.normal_buffer[-i:]
                    if tag.startswith(suffix):
                        hold_len = i
                
                if hold_len > 0:
                    yield_str = self.normal_buffer[:-hold_len]
                    self.normal_buffer = self.normal_buffer[-hold_len:]
                    return yield_str, None, None
                else:
                    yield_str = self.normal_buffer
                    self.normal_buffer = ""
                    return yield_str, None, None
        else:
            tool_res = self._feed_tool(chunk)
            if isinstance(tool_res, tuple):
                tool_delta, finish_reason, after_normal = tool_res
                return after_normal, tool_delta, finish_reason
            else:
                return "", tool_res, None

    def _feed_tool(self, chunk):
        self.tool_buffer += chunk
        
        if "[/tool_calls]" in self.tool_buffer:
            parts = self.tool_buffer.split("[/tool_calls]", 1)
            tool_content = parts[0]
            self.done = True
            self.tool_buffer = tool_content
            tool_delta = self._parse_tool_delta()
            if tool_delta is None:
                tool_delta = []
            normal_delta = parts[1] if len(parts) > 1 else ""
            return tool_delta, "tool_calls", normal_delta
        else:
            tag = "[/tool_calls]"
            hold_len = 0
            for i in range(1, len(tag)):
                suffix = self.tool_buffer[-i:]
                if tag.startswith(suffix):
                    hold_len = i
            if hold_len > 0:
                active_content = self.tool_buffer[:-hold_len]
                return self._parse_tool_delta(limit=len(active_content))
            else:
                return self._parse_tool_delta()

    def _parse_tool_delta(self, limit=None):
        content_to_parse = self.tool_buffer if limit is None else self.tool_buffer[:limit]
        tool_delta = []
        
        if not self.yielded_name:
            match = re.search(r'["\']name["\']\s*:\s*["\']([^"\']+)["\']', content_to_parse)
            if match:
                self.func_name = match.group(1)
                self.yielded_name = True
                tool_delta.append({
                    "index": 0,
                    "id": self.tool_call_id,
                    "type": "function",
                    "function": {
                        "name": self.func_name,
                        "arguments": ""
                    }
                })
        
        if self.yielded_name and self.arg_start_idx == -1:
            match_args = re.search(r'["\']arguments["\']\s*:\s*', content_to_parse)
            if match_args:
                end_pos = match_args.end()
                sub = content_to_parse[end_pos:]
                brace_match = re.search(r'[\{\[]', sub)
                if brace_match:
                    self.arg_start_idx = end_pos + brace_match.start()
                    self.arg_yielded_len = 0
                    self.brace_count = 0
                    
        if self.arg_start_idx != -1:
            args_sub = content_to_parse[self.arg_start_idx:]
            new_chars = args_sub[self.arg_yielded_len:]
            for char in new_chars:
                if char in ('{', '['):
                    self.brace_count += 1
                elif char in ('}', ']'):
                    self.brace_count -= 1
                
                if self.brace_count >= 0:
                    tool_delta.append({
                        "index": 0,
                        "function": {
                            "arguments": char
                        }
                    })
                    self.arg_yielded_len += 1
                else:
                    break
                    
        return tool_delta if tool_delta else None

def format_tool_instructions(tools):
    if not tools:
        return ""
    
    tools_formatted = []
    for tool in tools:
        if tool.get("type") == "function":
            tools_formatted.append(tool["function"])
            
    tools_json = json.dumps(tools_formatted, indent=2)
    
    instructions = f"""
You have access to the following functions (tools) that you can call if needed:
{tools_json}

If you decide to call any function(s), you MUST wrap the function call(s) inside a `[tool_calls]` block.
The content inside the `[tool_calls]` block MUST be a valid JSON array of objects, where each object has:
- "name": string (the name of the function to call)
- "arguments": object (the arguments to pass to the function)

Example output if you want to call a function:
[tool_calls]
[
  {{
    "name": "get_current_weather",
    "arguments": {{
      "location": "San Francisco, CA"
    }}
  }}
]
[/tool_calls]

If you call a function, do not output any other text or reasoning inside the `[tool_calls]` block. You may provide regular response text outside of the block if necessary, but keep it brief.
If no function calls are needed, respond normally without any `[tool_calls]` tag.
"""
    return instructions

def extract_prompt(messages, tools=None):
    """Extract system prompt and user message, combining them if system prompt exists."""
    system_messages = [m["content"] for m in messages if m.get("role") == "system"]
    user_messages = [m["content"] for m in messages if m.get("role") == "user"]
    
    system_prompt = "\n".join(system_messages) if system_messages else ""
    user_message = user_messages[-1] if user_messages else (messages[-1]["content"] if messages else "")
    
    if tools:
        tool_instructions = format_tool_instructions(tools)
        if system_prompt:
            system_prompt = f"{system_prompt}\n\n{tool_instructions}"
        else:
            system_prompt = tool_instructions
            
    if system_prompt:
        return f"System Prompt: {system_prompt}\n\nUser Message: {user_message}"
    return user_message

def chat_non_streaming(messages, model_type="default", thinking_enabled=True, tools=None):
    """Non-streaming chat"""
    prompt = extract_prompt(messages, tools)
    chat = DeepSeekChat(DS_SESSION_ID, AUTHORIZATION_TOKEN)
    res = chat.send_message(prompt, printing=False, thinking_enabled=thinking_enabled, search_enabled=False, model_type=model_type)
    if res and res.get("ok"):
        content = res.get("content", {})
        response_text = content.get("response", "")
        
        if tools and "[tool_calls]" in response_text and "[/tool_calls]" in response_text:
            parts = response_text.split("[tool_calls]", 1)
            text_before = parts[0].strip()
            rest = parts[1].split("[/tool_calls]", 1)
            tool_json = rest[0].strip()
            text_after = rest[1].strip() if len(rest) > 1 else ""
            
            try:
                tool_calls_raw = json.loads(tool_json)
                tool_calls = []
                for i, tc in enumerate(tool_calls_raw):
                    tool_calls.append({
                        "id": f"call_{tc.get('name', '')}_{i}",
                        "type": "function",
                        "function": {
                            "name": tc.get("name"),
                            "arguments": json.dumps(tc.get("arguments"))
                        }
                    })
                
                combined_content = (text_before + "\n" + text_after).strip()
                return {
                    "content": combined_content if combined_content else None,
                    "tool_calls": tool_calls,
                    "finish_reason": "tool_calls"
                }
            except Exception:
                pass
                
        return {
            "content": response_text,
            "tool_calls": None,
            "finish_reason": "stop"
        }
    else:
        error_msg = res.get("content") if res else "Unknown error"
        return {
            "content": f"Error: {error_msg}",
            "tool_calls": None,
            "finish_reason": "error"
        }

def chat_streaming(messages, model_type="default", thinking_enabled=True, tools=None):
    """Streaming chat using SSE and thread-safe queue"""
    prompt = extract_prompt(messages, tools)
    q = queue.Queue()
    
    def on_token(mode, text):
        q.put((mode, text))
        
    def run_chat():
        try:
            chat = DeepSeekChat(DS_SESSION_ID, AUTHORIZATION_TOKEN)
            res = chat.send_message(prompt, printing=on_token, thinking_enabled=thinking_enabled, search_enabled=False, model_type=model_type)
            if not res or not res.get("ok"):
                error_msg = res.get("content") if res else "Unknown error"
                q.put(("error", f"Error: {error_msg}"))
        except Exception as e:
            q.put(("error", str(e)))
        finally:
            q.put(None)
            
    t = threading.Thread(target=run_chat)
    t.start()
    
    parser = StreamingToolParser() if tools else None
    
    while True:
        item = q.get()
        if item is None:
            break
            
        mode, text = item
        if mode == "error":
            data_str = json.dumps({'choices': [{'delta': {'content': f"\n[Error: {text}]"}}]})
            yield "data: " + data_str + "\n\n"
            break
        elif mode == "THINK":
            data_str = json.dumps({'choices': [{'delta': {'reasoning_content': text}}]})
            yield "data: " + data_str + "\n\n"
        elif mode == "RESPONSE":
            if parser:
                normal_delta, tool_delta, finish_reason = parser.feed(text)
                if normal_delta:
                    data_str = json.dumps({'choices': [{'delta': {'content': normal_delta}}]})
                    yield "data: " + data_str + "\n\n"
                if tool_delta:
                    data_str = json.dumps({'choices': [{'delta': {'tool_calls': tool_delta}}]})
                    yield "data: " + data_str + "\n\n"
                if finish_reason:
                    data_str = json.dumps({'choices': [{'delta': {}, 'finish_reason': finish_reason}]})
                    yield "data: " + data_str + "\n\n"
            else:
                data_str = json.dumps({'choices': [{'delta': {'content': text}}]})
                yield "data: " + data_str + "\n\n"
                
    yield "data: [DONE]\n\n"

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/v1/chat/completions", methods=["POST"])
def chat_completions():
    data = request.json
    
    messages = data.get("messages", [])
    stream = data.get("stream", False)
    tools = data.get("tools")
    
    # Determine model - default to DeepSeek V3.
    model = data.get("model", "deepseek-v3")
    
    # Determine DeepSeek web model flags based on model name.
    model_type, thinking_enabled = get_model_config(model)
    
    if stream:
        return Response(
            stream_with_context(chat_streaming(messages, model_type, thinking_enabled, tools)),
            mimetype='text/event-stream',
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
            }
        )
    else:
        result = chat_non_streaming(messages, model_type, thinking_enabled, tools)
        
        message_dict = {
            "role": "assistant",
            "content": result.get("content")
        }
        if result.get("tool_calls"):
            message_dict["tool_calls"] = result["tool_calls"]
            
        content_len = len(result.get("content").split()) if result.get("content") else 0
        
        return jsonify({
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": message_dict,
                "finish_reason": result.get("finish_reason", "stop")
            }],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": content_len,
                "total_tokens": content_len
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
            },
            {
                "id": "deepseek-v4",
                "object": "model",
                "created": 1704067200,
                "owned_by": "deepseek",
                "description": "DeepSeek V4 - Expert model without extended thinking"
            },
            {
                "id": "deepseek-r4",
                "object": "model",
                "created": 1704067200,
                "owned_by": "deepseek",
                "description": "DeepSeek R4 - Expert reasoning model with extended thinking"
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
