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
    # Since sessions are deleted after every request to keep the sidebar clean,
    # we bypass the session cache to avoid using stale sessions and prevent 
    # unnecessary round-trip failures.
    return None, None

def save_session(messages, response_content, chat_session_id, parent_message_id, tool_calls=None):
    return

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

TOOL_WRAP_HINT = (
    "\n\n### SYSTEM: TOOL CALLING PROTOCOL (MANDATORY) ###\n"
    "If tool execution is required, you MUST adhere to this EXACT protocol. No exceptions.\n\n"
    "1. OUTPUT RESTRICTION: Your response MUST contain ONLY the [ToolCalls] block. Conversational filler, preambles, or concluding remarks are STRICTLY PROHIBITED.\n"
    "2. WRAPPING LOGIC: Every parameter value MUST be enclosed in a markdown code block. Use 3 backticks (```) by default. If the value contains backticks, the outer fence MUST be longer than any sequence inside (e.g., ````).\n"
    "3. TAG SYMMETRY: All tags MUST be balanced and closed in the exact reverse order of opening. Incomplete or unclosed blocks are strictly prohibited.\n\n"
    "REQUIRED SYNTAX:\n"
    "[ToolCalls]\n"
    "[Call:tool_name]\n"
    "[CallParameter:parameter_name]\n"
    "```\n"
    "value\n"
    "```\n"
    "[/CallParameter]\n"
    "[/Call]\n"
    "[/ToolCalls]\n\n"
    "CRITICAL: Do NOT mix natural language with protocol tags. Either respond naturally OR provide the protocol block alone. There is no middle ground."
)

TOOL_BLOCK_RE = re.compile(
    r"\\?\[ToolCalls\\?](.*?)\\?\[\\?/ToolCalls\\?]",
    re.DOTALL | re.IGNORECASE,
)
TOOL_CALL_RE = re.compile(
    r"\\?\[Call\\?:(?P<name>[^]]+)\\?](?P<body>.*?)\\?\[\\?/Call\\?]",
    re.DOTALL | re.IGNORECASE,
)
TAGGED_ARG_RE = re.compile(
    r"\\?\[CallParameter\\?:(?P<name>[^]]+)\\?](?P<body>.*?)\\?\[\\?/CallParameter\\?]",
    re.DOTALL | re.IGNORECASE,
)
COMMONMARK_UNESCAPE_RE = re.compile(r"\\([!\"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~])")
PARAM_FENCE_RE = re.compile(r"^(?P<fence>`{3,})")

_START_PATTERNS = {
    "TOOL": r"\\?\[ToolCalls\\?]",
    "ORPHAN": r"\\?\[Call\\?:[^]]+\\?]",
    "ARG": r"\\?\[CallParameter\\?:[^]]+\\?]",
}
_PROTOCOL_ENDS = r"\\?\[\\?/(?:ToolCalls|Call|CallParameter)\\?]"

_master_parts = [f"(?P<{name}_START>{pattern})" for name, pattern in _START_PATTERNS.items()]
_master_parts.extend((f"(?P<PROTOCOL_EXIT>{_PROTOCOL_ENDS})",))

STREAM_MASTER_RE = re.compile("|".join(_master_parts), re.IGNORECASE)
STREAM_TAIL_RE = re.compile(
    r"(?:\\|\\?\[[^]]*)$",
    re.IGNORECASE,
)

def unescape_text(s: str) -> str:
    """Remove CommonMark backslash escapes from LLM-generated text."""
    return COMMONMARK_UNESCAPE_RE.sub(r"\1", s) if s else ""

def strip_markdown_fence(s: str) -> str:
    s = s.strip()
    if not s:
        return ""
    match = PARAM_FENCE_RE.match(s)
    if not match or not s.endswith(match.group("fence")):
        return s
    fence = match.group("fence")
    lines = s.splitlines()
    if len(lines) >= 3 and lines[-1].strip() == fence:
        return "\n".join(lines[1:-1])
    return s[len(fence) : -len(fence)].strip()

def _parse_tool_argument_value(raw_value: str):
    value = strip_markdown_fence(raw_value)
    if not value:
        return ""
    try:
        return json.loads(value)
    except Exception:
        return value

def strip_system_hints(text: str) -> str:
    if not text:
        return text
    t_unescaped = unescape_text(text)
    cleaned = t_unescaped.replace(TOOL_WRAP_HINT, "")
    cleaned = TOOL_BLOCK_RE.sub("", cleaned)
    cleaned = TOOL_CALL_RE.sub("", cleaned)
    cleaned = TAGGED_ARG_RE.sub("", cleaned)
    return cleaned

def extract_tool_calls(text: str):
    if not text:
        return text, []

    tool_calls = []

    def _create_tool_call(name: str, raw_args: str) -> None:
        if not name:
            return
        name = unescape_text(name.strip())
        raw_args = unescape_text(raw_args)

        arg_matches = TAGGED_ARG_RE.findall(raw_args)
        if arg_matches:
            args_dict = {
                arg_name.strip(): _parse_tool_argument_value(arg_value)
                for arg_name, arg_value in arg_matches
            }
            arguments = json.dumps(args_dict)
        else:
            arguments = "{}"

        index = len(tool_calls)
        call_id = f"call_{name}_{index}_{int(time.time())}"

        tool_calls.append({
            "id": call_id,
            "type": "function",
            "function": {
                "name": name,
                "arguments": arguments
            }
        })

    for match in TOOL_CALL_RE.finditer(text):
        _create_tool_call(match.group(1), match.group(2))

    cleaned = strip_system_hints(text).strip()
    return cleaned, tool_calls

def extract_tool_calls_from_text(response_text, tools):
    if not tools:
        return False, None, response_text
    
    cleaned, tool_calls = extract_tool_calls(response_text)
    if tool_calls:
        return True, tool_calls, (cleaned if cleaned else None)
    return False, None, response_text

class StreamingOutputFilter:
    def __init__(self):
        self.buffer = ""
        self.stack = ["NORMAL"]

    @property
    def state(self):
        return self.stack[-1]

    def _is_outputting(self) -> bool:
        if self.state == "POST_BLOCK":
            return False
        return self.state == "NORMAL"

    def process(self, chunk: str) -> str:
        self.buffer += chunk
        output = []

        while self.buffer:
            if self.state == "POST_BLOCK":
                stripped = self.buffer.lstrip()
                if not stripped:
                    break
                self.buffer = stripped
                self.stack[-1] = "NORMAL"

            match = STREAM_MASTER_RE.search(self.buffer)
            if not match:
                if tail_match := STREAM_TAIL_RE.search(self.buffer):
                    yield_len = len(self.buffer) - len(tail_match.group(0))
                    if yield_len > 0:
                        if self._is_outputting():
                            output.append(self.buffer[:yield_len])
                        self.buffer = self.buffer[yield_len:]
                else:
                    if self._is_outputting():
                        output.append(self.buffer)
                    self.buffer = ""
                break

            start, end = match.span()
            matched_group = match.lastgroup
            pre_text = self.buffer[:start]

            if self._is_outputting():
                output.append(pre_text)

            if matched_group and matched_group.endswith("_START"):
                m_type = matched_group.split("_")[0]
                self.stack.append(f"IN_{m_type}")
            elif matched_group in ("PROTOCOL_EXIT",):
                if len(self.stack) > 1:
                    self.stack.pop()
                else:
                    self.stack = ["NORMAL"]

                if self.state == "NORMAL" and matched_group in ("PROTOCOL_EXIT",):
                    self.stack[-1] = "POST_BLOCK"

            self.buffer = self.buffer[end:]

        return "".join(output)

    def flush(self) -> str:
        res = ""
        if self._is_outputting():
            res = self.buffer
            if tail_match := STREAM_TAIL_RE.search(res):
                res = res[: -len(tail_match.group(0))]

        self.buffer = ""
        self.stack = ["NORMAL"]
        return strip_system_hints(res)

class StreamingToolParser:
    def __init__(self):
        self.filter = StreamingOutputFilter()
        
    def feed(self, chunk):
        return self.filter.process(chunk)
        
    def flush(self):
        return self.filter.flush()

def format_tool_instructions(tools):
    if not tools:
        return ""

    lines = [
        "SYSTEM INTERFACE: You have access to the following technical tools. You MUST invoke them when necessary to fulfill the request, strictly adhering to the provided JSON schemas."
    ]

    for tool in tools:
        if tool.get("type") == "function":
            function = tool["function"]
            description = function.get("description") or "No description provided."
            lines.append(f"Tool `{function['name']}`: {description}")
            if function.get("parameters"):
                schema_text = json.dumps(function["parameters"], sort_keys=True)
                lines.append(f"Arguments JSON schema: {schema_text}")
            else:
                lines.append("Arguments JSON schema: {}")

    lines.append(TOOL_WRAP_HINT)
    return "\n".join(lines)

def extract_prompt(messages, tools=None):
    """Extract system prompt and user message, combining them if system prompt exists."""
    system_messages = [m["content"] for m in messages if m.get("role") == "system"]
    chat_messages = [m for m in messages if m.get("role") != "system"]
    
    if len(chat_messages) <= 1:
        m = chat_messages[-1] if chat_messages else None
        if m and m.get("role") == "tool":
            tool_name = m.get("name") or "unknown"
            user_message = f"[Tool Response ({tool_name})]: [ToolResults]\n[Result:{tool_name}]\n[ToolResult]\n{m.get('content') or ''}\n[/ToolResult]\n[/Result]\n[/ToolResults]"
        else:
            user_message = m.get("content") or "" if m else ""
    else:
        formatted_turns = []
        
        # Build tool_id to name mapping
        tool_id_to_name = {}
        for m in chat_messages:
            if m.get("role") == "assistant" and m.get("tool_calls"):
                for tc in m["tool_calls"]:
                    tool_id_to_name[tc.get("id")] = tc.get("function", {}).get("name")
                    
        for m in chat_messages[:-1]:
            role = m.get("role")
            content = m.get("content") or ""
            
            # Truncate content of individual history messages if they are too long
            if len(content) > 1500:
                content = content[:1500] + "\n... [Content Truncated for Length] ..."
                
            if role == "assistant" and m.get("tool_calls"):
                # Format previous tool calls using the PascalCase protocol
                tool_blocks = []
                for call in m["tool_calls"]:
                    func = call.get("function", {})
                    params_text = func.get("arguments", "").strip()
                    formatted_params = ""
                    if params_text:
                        try:
                            parsed_params = json.loads(params_text)
                            if isinstance(parsed_params, dict):
                                for k, v in parsed_params.items():
                                    val_str = v if isinstance(v, str) else json.dumps(v)
                                    formatted_params += f"[CallParameter:{k}]\n```\n{val_str}\n```\n[/CallParameter]\n"
                            else:
                                formatted_params += f"```\n{params_text}\n```\n"
                        except Exception:
                            formatted_params += f"```\n{params_text}\n```\n"
                    tool_blocks.append(f"[Call:{func.get('name')}]\n{formatted_params}[/Call]")
                
                tool_section = "[ToolCalls]\n" + "\n".join(tool_blocks) + "\n[/ToolCalls]"
                prefix = f"{content}\n" if content else ""
                formatted_turns.append(f"[Assistant]: {prefix}{tool_section}")
                
            elif role == "tool":
                tool_name = m.get("name") or tool_id_to_name.get(m.get("tool_call_id")) or "unknown"
                wrapped_content = f"[ToolResults]\n[Result:{tool_name}]\n[ToolResult]\n{content}\n[/ToolResult]\n[/Result]\n[/ToolResults]"
                formatted_turns.append(f"[Tool Response ({tool_name})]: {wrapped_content}")
            elif role == "user":
                formatted_turns.append(f"[User]: {content}")
            elif role == "assistant":
                formatted_turns.append(f"[Assistant]: {content}")
                
        # Format the last message
        last_msg_obj = chat_messages[-1]
        last_msg_role = last_msg_obj.get("role")
        last_msg_content = last_msg_obj.get("content") or ""
        
        if last_msg_role == "tool":
            tool_name = last_msg_obj.get("name") or tool_id_to_name.get(last_msg_obj.get("tool_call_id")) or "unknown"
            formatted_last = f"[Tool Response ({tool_name})]: [ToolResults]\n[Result:{tool_name}]\n[ToolResult]\n{last_msg_content}\n[/ToolResult]\n[/Result]\n[/ToolResults]"
        elif last_msg_role == "user":
            formatted_last = f"[User]: {last_msg_content}"
        else:
            formatted_last = f"[{last_msg_role.capitalize()}]: {last_msg_content}"
        
        # Keep most recent turns under a total length limit to fit into DeepSeek web UI limits
        allowed_history_chars = 4000
        pruned_turns = []
        total_len = 0
        for turn in reversed(formatted_turns):
            if total_len + len(turn) + 1 <= allowed_history_chars:
                pruned_turns.insert(0, turn)
                total_len += len(turn) + 1
            else:
                break
                
        if len(pruned_turns) < len(formatted_turns):
            pruned_turns.insert(0, "... [Older Conversation History Omitted to Prevent Length Error] ...")
            
        history_str = "\n".join(pruned_turns)
        user_message = f"Conversation History:\n{history_str}\n\n{formatted_last}"
        
    system_prompt = "\n".join(system_messages) if system_messages else ""
    
    if tools:
        tool_instructions = format_tool_instructions(tools)
        if system_prompt:
            system_prompt = f"{system_prompt}\n\n{tool_instructions}"
        else:
            system_prompt = tool_instructions
            
    if system_prompt:
        return f"System Prompt:\n{system_prompt}\n\nUser Message:\n{user_message}"
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
        
        is_tool_call, tool_calls, final_content = extract_tool_calls_from_text(response_text, tools)
                
        if is_tool_call:
            save_session(messages, final_content, chat.chat_session_id, chat.parent_message_id, tool_calls=tool_calls)
            return {
                "content": final_content,
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
    raw_response_text = ""
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
            raw_response_text += text
            if parser:
                normal_delta = parser.feed(text)
                if normal_delta:
                    accumulated_content += normal_delta
                    yield {"type": "content", "text": normal_delta}
            else:
                accumulated_content += text
                yield {"type": "content", "text": text}
                
    if parser:
        remaining_content = parser.flush()
        if remaining_content:
            accumulated_content += remaining_content
            yield {"type": "content", "text": remaining_content}
            
    if tools:
        cleaned, tool_calls = extract_tool_calls(raw_response_text)
        if tool_calls:
            formatted_calls = []
            for idx, call in enumerate(tool_calls):
                formatted_calls.append({
                    "index": idx,
                    "id": call["id"],
                    "type": "function",
                    "function": {
                        "name": call["function"]["name"],
                        "arguments": call["function"]["arguments"]
                    }
                })
            yield {"type": "tool_calls", "calls": formatted_calls}
            yield {"type": "finish_reason", "reason": "tool_calls"}
            
            if final_session_id and final_parent_id:
                save_session(messages, cleaned, final_session_id, final_parent_id, tool_calls=tool_calls)
        else:
            if final_session_id and final_parent_id:
                save_session(messages, accumulated_content, final_session_id, final_parent_id)
    else:
        if final_session_id and final_parent_id:
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
        
    if not DS_SESSION_ID or not AUTHORIZATION_TOKEN:
        return JSONResponse({
            "error": {
                "message": "DeepSeek API Credentials are not configured. Ensure a 'tokens' file is present or DS_SESSION_ID/AUTHORIZATION_TOKEN env variables are set.",
                "type": "authentication_error",
                "param": None,
                "code": "invalid_api_key"
            }
        }, status_code=401)
    
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
    
    env_host = os.environ.get("HOST", "0.0.0.0")
    env_port = int(os.environ.get("PORT", "8000"))
    
    parser = argparse.ArgumentParser(description="DeepSeek API Server")
    parser.add_argument("--host", default=env_host, help="Host to bind")
    parser.add_argument("--port", type=int, default=env_port, help="Port to bind")
    args = parser.parse_args()
    
    print(f"Starting DeepSeek API Server on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
