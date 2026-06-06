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
import hashlib
from io import StringIO
from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.concurrency import run_in_threadpool
from DeepSeekAPI import DeepSeekChat

app = FastAPI(title="DeepSeek API Server")

# Thread-safe multi-turn session cache
session_cache = {}
cache_lock = threading.Lock()

def get_messages_hash(messages):
    normalized = []
    for m in messages:
        normalized.append({
            "role": m.get("role"),
            "content": m.get("content"),
            "tool_calls": m.get("tool_calls")
        })
    return hashlib.sha256(json.dumps(normalized, sort_keys=True).encode('utf-8')).hexdigest()

def lookup_session(messages):
    if not messages or len(messages) <= 1:
        return None, None
    prefix = messages[:-1]
    prefix_hash = get_messages_hash(prefix)
    with cache_lock:
        return session_cache.get(prefix_hash, (None, None))

def save_session(messages, response_content, chat_session_id, parent_message_id, tool_calls=None):
    if not chat_session_id or not parent_message_id:
        return
    assistant_msg = {"role": "assistant", "content": response_content}
    if tool_calls:
        assistant_msg["tool_calls"] = tool_calls
    new_history = list(messages) + [assistant_msg]
    new_hash = get_messages_hash(new_history)
    with cache_lock:
        if len(session_cache) >= 2000:
            first_key = next(iter(session_cache))
            session_cache.pop(first_key, None)
        session_cache[new_hash] = (chat_session_id, parent_message_id)

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

try:
    DS_SESSION_ID, AUTHORIZATION_TOKEN = get_tokens()
except Exception as e:
    print(f"Warning: {e}")
    DS_SESSION_ID, AUTHORIZATION_TOKEN = None, None

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
    chat_messages = [m for m in messages if m.get("role") != "system"]
    
    if len(chat_messages) <= 1:
        user_message = chat_messages[-1]["content"] if chat_messages else ""
    else:
        formatted_turns = []
        for m in chat_messages:
            role = m.get("role")
            content = m.get("content") or ""
            if role == "assistant" and m.get("tool_calls"):
                formatted_turns.append(f"Assistant (Tool Calls): {json.dumps(m['tool_calls'])}")
            elif role == "tool":
                formatted_turns.append(f"Tool Response ({m.get('name', '')}): {content}")
            elif role == "user":
                formatted_turns.append(f"User: {content}")
            elif role == "assistant":
                formatted_turns.append(f"Assistant: {content}")
        user_message = "\n\n".join(formatted_turns)
        
    system_prompt = "\n".join(system_messages) if system_messages else ""
    
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
    if not DS_SESSION_ID or not AUTHORIZATION_TOKEN:
        raise ValueError("Tokens not initialized. Ensure DS_SESSION_ID and AUTHORIZATION_TOKEN are set correctly.")

    chat_session_id, parent_message_id = lookup_session(messages)
    chat = DeepSeekChat(DS_SESSION_ID, AUTHORIZATION_TOKEN)
    
    if chat_session_id and parent_message_id:
        chat.chat_session_id = chat_session_id
        chat.parent_message_id = parent_message_id
        prompt = messages[-1].get("content") or ""
    else:
        prompt = extract_prompt(messages, tools)
        
    res = chat.send_message(prompt, printing=False, thinking_enabled=thinking_enabled, search_enabled=False, model_type=model_type)
    chat.delete_all_sessions()
    
    if not res or not res.get("ok"):
        error_content = str(res.get("content") if res else "")
        if "invalid chat session id" in error_content:
            prefix = messages[:-1]
            prefix_hash = get_messages_hash(prefix)
            with cache_lock:
                session_cache.pop(prefix_hash, None)
            chat.chat_session_id = None
            chat.parent_message_id = None
            prompt = extract_prompt(messages, tools)
            res = chat.send_message(prompt, printing=False, thinking_enabled=thinking_enabled, search_enabled=False, model_type=model_type)
            chat.delete_all_sessions()
            
    if res and res.get("ok"):
        content = res.get("content", {})
        response_text = content.get("response", "")
        
        is_tool_call = False
        tool_calls = None
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
                is_tool_call = True
                combined_content = (text_before + "\n" + text_after).strip()
                response_text = combined_content if combined_content else None
            except Exception:
                pass
                
        if is_tool_call:
            save_session(messages, response_text, chat.chat_session_id, chat.parent_message_id, tool_calls=tool_calls)
            return {
                "content": response_text,
                "tool_calls": tool_calls,
                "finish_reason": "tool_calls"
            }
        else:
            save_session(messages, response_text, chat.chat_session_id, chat.parent_message_id)
            return {
                "content": response_text,
                "tool_calls": None,
                "finish_reason": "stop"
            }
    else:
        error_msg = res.get("content") if res else "Unknown error"
        raise ValueError(error_msg)

def chat_streaming(messages, model_type="default", thinking_enabled=True, tools=None):
    """Streaming generator yielding structured dictionaries for modular parsing"""
    if not DS_SESSION_ID or not AUTHORIZATION_TOKEN:
        yield {"type": "error", "error": "Tokens not initialized. Ensure DS_SESSION_ID and AUTHORIZATION_TOKEN are set correctly."}
        return

    chat_session_id, parent_message_id = lookup_session(messages)
    chat = DeepSeekChat(DS_SESSION_ID, AUTHORIZATION_TOKEN)
    
    if chat_session_id and parent_message_id:
        chat.chat_session_id = chat_session_id
        chat.parent_message_id = parent_message_id
        prompt = messages[-1].get("content") or ""
    else:
        prompt = extract_prompt(messages, tools)
        
    q = queue.Queue()
    
    def on_token(mode, text):
        q.put((mode, text))
        
    def run_chat():
        try:
            res = chat.send_message(prompt, printing=on_token, thinking_enabled=thinking_enabled, search_enabled=False, model_type=model_type)
            if not res or not res.get("ok"):
                error_msg = res.get("content") if res else "Unknown error"
                if "invalid chat session id" in str(error_msg):
                    prefix = messages[:-1]
                    prefix_hash = get_messages_hash(prefix)
                    with cache_lock:
                        session_cache.pop(prefix_hash, None)
                    chat.chat_session_id = None
                    chat.parent_message_id = None
                    fallback_prompt = extract_prompt(messages, tools)
                    res = chat.send_message(fallback_prompt, printing=on_token, thinking_enabled=thinking_enabled, search_enabled=False, model_type=model_type)
                    if not res or not res.get("ok"):
                        error_msg = res.get("content") if res else "Unknown error"
                        q.put(("error", error_msg))
                else:
                    q.put(("error", error_msg))
        except Exception as e:
            q.put(("error", str(e)))
        finally:
            chat.delete_all_sessions()
            q.put(("done_session", (chat.chat_session_id, chat.parent_message_id)))
            q.put(None)
            
    t = threading.Thread(target=run_chat)
    t.start()
    
    parser = StreamingToolParser() if tools else None
    accumulated_content = ""
    final_session_id = None
    final_parent_id = None
    
    while True:
        item = q.get()
        if item is None:
            break
            
        if isinstance(item, tuple) and item[0] == "done_session":
            final_session_id, final_parent_id = item[1]
            continue
            
        mode, text = item
        if mode == "error":
            yield {"type": "error", "error": text}
            break
        elif mode == "THINK":
            yield {"type": "think", "text": text}
        elif mode == "RESPONSE":
            if parser:
                normal_delta, tool_delta, finish_reason = parser.feed(text)
                if normal_delta:
                    accumulated_content += normal_delta
                    yield {"type": "content", "text": normal_delta}
                if tool_delta:
                    yield {"type": "tool_calls", "calls": tool_delta}
                if finish_reason:
                    yield {"type": "finish_reason", "reason": finish_reason}
            else:
                accumulated_content += text
                yield {"type": "content", "text": text}
                
    if final_session_id and final_parent_id:
        if parser and parser.done:
            tool_calls = [{
                "id": parser.tool_call_id,
                "type": "function",
                "function": {
                    "name": parser.func_name,
                    "arguments": parser.tool_buffer
                }
            }]
            save_session(messages, None, final_session_id, final_parent_id, tool_calls=tool_calls)
        else:
            save_session(messages, accumulated_content, final_session_id, final_parent_id)

def format_chunk(item):
    if item["type"] == "think":
        data_str = json.dumps({'choices': [{'delta': {'reasoning_content': item["text"]}}]})
        yield "data: " + data_str + "\n\n"
    elif item["type"] == "content":
        data_str = json.dumps({'choices': [{'delta': {'content': item["text"]}}]})
        yield "data: " + data_str + "\n\n"
    elif item["type"] == "tool_calls":
        data_str = json.dumps({'choices': [{'delta': {'tool_calls': item["calls"]}}]})
        yield "data: " + data_str + "\n\n"
    elif item["type"] == "finish_reason":
        data_str = json.dumps({'choices': [{'delta': {}, 'finish_reason': item["reason"]}]})
        yield "data: " + data_str + "\n\n"
    elif item["type"] == "error":
        data_str = json.dumps({'choices': [{'delta': {'content': f"\n[Error: {item['error']}]"}}]})
        yield "data: " + data_str + "\n\n"

@app.get("/", response_class=HTMLResponse)
async def index():
    try:
        with open("templates/index.html", "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error loading index.html: {e}"

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    try:
        data = await request.json()
    except Exception:
        return JSONResponse({
            "error": {
                "message": "Malformed JSON request body.",
                "type": "invalid_request_error",
                "param": None,
                "code": "bad_request"
            }
        }, status_code=400)
    
    messages = data.get("messages", [])
    stream = data.get("stream", False)
    tools = data.get("tools")
    model = data.get("model", "deepseek-v3")
    
    # 100% OpenAI input parameter validations
    if not isinstance(messages, list):
        return JSONResponse({
            "error": {
                "message": "'messages' must be a list of objects.",
                "type": "invalid_request_error",
                "param": "messages",
                "code": "invalid_value"
            }
        }, status_code=400)
        
    if not messages:
        return JSONResponse({
            "error": {
                "message": "'messages' must not be empty.",
                "type": "invalid_request_error",
                "param": "messages",
                "code": "empty_value"
            }
        }, status_code=400)
        
    # Check for authorization tokens
    if not DS_SESSION_ID or not AUTHORIZATION_TOKEN:
        return JSONResponse({
            "error": {
                "message": "DeepSeek API Credentials are not configured. Ensure a 'tokens' file is present or DS_SESSION_ID/AUTHORIZATION_TOKEN env variables are set.",
                "type": "authentication_error",
                "param": None,
                "code": "invalid_api_key"
            }
        }, status_code=401)
    
    # Determine DeepSeek web model flags based on model name.
    model_type, thinking_enabled = get_model_config(model)
    
    if stream:
        try:
            gen = chat_streaming(messages, model_type, thinking_enabled, tools)
            try:
                first_item = next(gen)
            except StopIteration:
                first_item = None
                
            if first_item and first_item["type"] == "error":
                error_msg = first_item["error"]
                status_code = 500
                error_type = "api_error"
                if "token" in error_msg.lower() or "auth" in error_msg.lower():
                    status_code = 401
                    error_type = "authentication_error"
                elif "rate limit" in error_msg.lower() or "429" in error_msg.lower():
                    status_code = 429
                    error_type = "rate_limit_error"
                return JSONResponse({
                    "error": {
                        "message": error_msg,
                        "type": error_type,
                        "param": None,
                        "code": None
                    }
                }, status_code=status_code)
                
            def sse_formatter(first, g):
                if first:
                    for chunk in format_chunk(first):
                        yield chunk
                for item in g:
                    for chunk in format_chunk(item):
                        yield chunk
                yield "data: [DONE]\n\n"
                
            return StreamingResponse(
                sse_formatter(first_item, gen),
                media_type='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'Connection': 'keep-alive',
                }
            )
        except Exception as e:
            return JSONResponse({
                "error": {
                    "message": str(e),
                    "type": "api_error",
                    "param": None,
                    "code": None
                }
            }, status_code=500)
    else:
        try:
            result = await run_in_threadpool(chat_non_streaming, messages, model_type, thinking_enabled, tools)
        except ValueError as e:
            error_msg = str(e)
            status_code = 500
            error_type = "api_error"
            if "token" in error_msg.lower() or "auth" in error_msg.lower():
                status_code = 401
                error_type = "authentication_error"
            elif "rate limit" in error_msg.lower() or "429" in error_msg.lower():
                status_code = 429
                error_type = "rate_limit_error"
            return JSONResponse({
                "error": {
                    "message": error_msg,
                    "type": error_type,
                    "param": None,
                    "code": None
                }
            }, status_code=status_code)
            
        message_dict = {
            "role": "assistant",
            "content": result.get("content")
        }
        if result.get("tool_calls"):
            message_dict["tool_calls"] = result["tool_calls"]
            
        content_len = len(result.get("content").split()) if result.get("content") else 0
        
        return JSONResponse({
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

@app.get("/v1/models")
async def list_models():
    return {
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
    }

@app.get("/health")
async def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import argparse
    import uvicorn
    parser = argparse.ArgumentParser(description="DeepSeek API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    args = parser.parse_args()
    
    print(f"Starting DeepSeek API Server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
