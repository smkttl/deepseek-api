import os
import sys
import json
import unittest

# Ensure the root of the project is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from server import app
from fastapi.testclient import TestClient

class TestDeepSeekServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokens_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../tokens"))
        if not os.path.exists(cls.tokens_path):
            raise unittest.SkipTest(f"Tokens file not found at {cls.tokens_path}. Skipping integration tests.")
        
        # Configure FastAPI TestClient
        cls.client = TestClient(app)

    def test_list_models(self):
        """Test GET /v1/models"""
        response = self.client.get("/v1/models")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["object"], "list")
        model_ids = [m["id"] for m in data["data"]]
        self.assertIn("deepseek-v3", model_ids)
        self.assertIn("deepseek-r1", model_ids)
        self.assertIn("deepseek-v4", model_ids)
        self.assertIn("deepseek-r4", model_ids)

    def test_health(self):
        """Test GET /health"""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_index_page(self):
        """Test GET / (web UI root)"""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"<!DOCTYPE html>", response.content)

    def test_chat_completions_non_streaming_v3(self):
        """Test POST /v1/chat/completions (non-streaming, deepseek-v3)"""
        payload = {
            "model": "deepseek-v3",
            "messages": [{"role": "user", "content": "Say hello"}],
            "stream": False
        }
        response = self.client.post("/v1/chat/completions", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["object"], "chat.completion")
        self.assertEqual(data["model"], "deepseek-v3")
        self.assertTrue(len(data["choices"]) > 0)
        content = data["choices"][0]["message"]["content"]
        self.assertIsNotNone(content)
        print("\nServer Non-Streaming Response:", content)

    def test_chat_completions_streaming_r1(self):
        """Test POST /v1/chat/completions (streaming, deepseek-r1 with reasoning)"""
        payload = {
            "model": "deepseek-r1",
            "messages": [{"role": "user", "content": "Say hello"}],
            "stream": True
        }
        
        with self.client.stream("POST", "/v1/chat/completions", json=payload) as response:
            self.assertEqual(response.status_code, 200)
            self.assertIn("text/event-stream", response.headers.get("content-type", ""))
            
            # Read the stream lines
            has_reasoning = False
            has_content = False
            
            for line in response.iter_lines():
                if not line:
                    continue
                line_str = line.strip()
                if not line_str or line_str == "data: [DONE]":
                    continue
                if line_str.startswith("data: "):
                    data = json.loads(line_str[6:])
                    delta = data["choices"][0]["delta"]
                    if "reasoning_content" in delta:
                        has_reasoning = True
                        print(delta["reasoning_content"], end="", flush=True)
                    if "content" in delta:
                        has_content = True
                        print(delta["content"], end="", flush=True)
                        
            print()
            self.assertTrue(has_content or has_reasoning, "Streaming returned no tokens.")

    def test_chat_completions_system_prompt_v3(self):
        """Test POST /v1/chat/completions with system prompt (deepseek-v3)"""
        payload = {
            "model": "deepseek-v3",
            "messages": [
                {"role": "system", "content": "You are a pirate. Always start your response with 'AHoy!'"},
                {"role": "user", "content": "Say hello"}
            ],
            "stream": False
        }
        response = self.client.post("/v1/chat/completions", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["object"], "chat.completion")
        self.assertEqual(data["model"], "deepseek-v3")
        self.assertTrue(len(data["choices"]) > 0)
        content = data["choices"][0]["message"]["content"]
        self.assertIsNotNone(content)
        print("\nSystem Prompt Response:", content)
        self.assertTrue(
            any(w in content.lower() for w in ["ahoy", "pirate", "matey", "scallywag", "buccaneer", "seas", "oy!", "sea", "sail"]),
            f"Model did not adhere to system prompt: {content}"
        )

    def test_chat_completions_function_calling_non_streaming(self):
        """Test POST /v1/chat/completions with function calling (non-streaming, deepseek-v3)"""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather of a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA"
                            }
                        },
                        "required": ["location"]
                    }
                }
            }
        ]
        payload = {
            "model": "deepseek-v3",
            "messages": [
                {"role": "user", "content": "What is the weather in Paris, France right now?"}
            ],
            "tools": tools,
            "stream": False
        }
        response = self.client.post("/v1/chat/completions", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        print("\nFunction Calling Non-Streaming Response:", json.dumps(data, indent=2))
        self.assertEqual(data["object"], "chat.completion")
        self.assertEqual(data["choices"][0]["finish_reason"], "tool_calls")
        message = data["choices"][0]["message"]
        self.assertIn("tool_calls", message)
        tool_calls = message["tool_calls"]
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0]["function"]["name"], "get_weather")
        
        args = json.loads(tool_calls[0]["function"]["arguments"])
        self.assertIn("location", args)
        self.assertTrue("paris" in args["location"].lower())

    def test_chat_completions_function_calling_streaming(self):
        """Test POST /v1/chat/completions with function calling (streaming, deepseek-v3)"""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the weather of a given location",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "The city and state, e.g. San Francisco, CA"
                            }
                        },
                        "required": ["location"]
                    }
                }
            }
        ]
        payload = {
            "model": "deepseek-v3",
            "messages": [
                {"role": "user", "content": "What is the weather in Paris, France right now?"}
            ],
            "tools": tools,
            "stream": True
        }
        
        with self.client.stream("POST", "/v1/chat/completions", json=payload) as response:
            self.assertEqual(response.status_code, 200)
            
            tool_calls = []
            finish_reason = None
            
            for line in response.iter_lines():
                if not line:
                    continue
                line_str = line.strip()
                if not line_str or line_str == "data: [DONE]":
                    continue
                if line_str.startswith("data: "):
                    data = json.loads(line_str[6:])
                    choice = data["choices"][0]
                    delta = choice["delta"]
                    if "tool_calls" in delta:
                        tool_calls.append(delta["tool_calls"])
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
                        
            print("\nFunction Calling Streaming tool_calls chunks:", tool_calls)
            self.assertEqual(finish_reason, "tool_calls")
            self.assertTrue(len(tool_calls) > 0)
            
            name = ""
            arguments = ""
            for tc_list in tool_calls:
                for tc in tc_list:
                    if "function" in tc:
                        if "name" in tc["function"]:
                            name = tc["function"]["name"]
                        if "arguments" in tc["function"]:
                            arguments += tc["function"]["arguments"]
                            
            self.assertEqual(name, "get_weather")
            args = json.loads(arguments)
            self.assertIn("location", args)
            self.assertTrue("paris" in args["location"].lower())

    def test_chat_completions_multi_turn(self):
        """Test POST /v1/chat/completions multi-turn conversation caching and fallback"""
        # Turn 1: Tell the model a secret color
        payload1 = {
            "model": "deepseek-v3",
            "messages": [{"role": "user", "content": "My favorite color is green. Please remember it."}],
            "stream": False
        }
        response1 = self.client.post("/v1/chat/completions", json=payload1)
        self.assertEqual(response1.status_code, 200)
        data1 = response1.json()
        content1 = data1["choices"][0]["message"]["content"]
        print("\nMulti-turn Turn 1 Response:", content1)
        
        # Sleep to avoid rate limiting
        import time
        time.sleep(3)
        
        # Turn 2: Ask about the secret color
        payload2 = {
            "model": "deepseek-v3",
            "messages": [
                {"role": "user", "content": "My favorite color is green. Please remember it."},
                {"role": "assistant", "content": content1},
                {"role": "user", "content": "What is my favorite color?"}
            ],
            "stream": False
        }
        response2 = self.client.post("/v1/chat/completions", json=payload2)
        self.assertEqual(response2.status_code, 200)
        data2 = response2.json()
        content2 = data2["choices"][0]["message"]["content"]
        print("Multi-turn Turn 2 Response:", content2)
        
        self.assertIn("green", content2.lower())

    def test_invalid_messages_parameter(self):
        """Test POST /v1/chat/completions with invalid messages type (not a list)"""
        payload = {
            "model": "deepseek-v3",
            "messages": "This is a string, not a list",
            "stream": False
        }
        response = self.client.post("/v1/chat/completions", json=payload)
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn("error", data)
        self.assertEqual(data["error"]["type"], "invalid_request_error")
        self.assertEqual(data["error"]["param"], "messages")
        self.assertEqual(data["error"]["code"], "invalid_value")

    def test_empty_messages_parameter(self):
        """Test POST /v1/chat/completions with empty messages list"""
        payload = {
            "model": "deepseek-v3",
            "messages": [],
            "stream": False
        }
        response = self.client.post("/v1/chat/completions", json=payload)
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn("error", data)
        self.assertEqual(data["error"]["type"], "invalid_request_error")
        self.assertEqual(data["error"]["param"], "messages")
        self.assertEqual(data["error"]["code"], "empty_value")

    def test_malformed_json_request(self):
        """Test POST /v1/chat/completions with malformed JSON request body"""
        response = self.client.post(
            "/v1/chat/completions",
            content="invalid raw string body",
            headers={"Content-Type": "application/json"}
        )
        self.assertEqual(response.status_code, 400)
        data = response.json()
        self.assertIn("error", data)
        self.assertEqual(data["error"]["type"], "invalid_request_error")
        self.assertEqual(data["error"]["code"], "bad_request")

    def tearDown(self):
        import time
        time.sleep(6)

if __name__ == "__main__":
    unittest.main()
