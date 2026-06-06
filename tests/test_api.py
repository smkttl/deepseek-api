import os
import sys
import unittest

# Ensure the root of the project is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from DeepSeekAPI import DeepSeekChat

class TestDeepSeekModels(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Locate tokens file in root directory relative to this test file
        cls.tokens_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../tokens"))
        if not os.path.exists(cls.tokens_path):
            raise unittest.SkipTest(f"Tokens file not found at {cls.tokens_path}. Skipping integration tests.")
            
        with open(cls.tokens_path) as f:
            lines = f.read().strip().split('\n')
            cls.ds_session_id = lines[0]
            cls.authorization_token = lines[1]

    def test_deepseek_v3(self):
        """Test deepseek-v3 (thinking disabled, default model type)."""
        chat = DeepSeekChat(self.ds_session_id, self.authorization_token)
        res = chat.send_message("Say hello", printing=False, thinking_enabled=False, model_type="default")
        
        self.assertTrue(res.get("ok"), f"Model deepseek-v3 failed: {res}")
        self.assertIn("content", res)
        content = res["content"]
        self.assertIn("response", content)
        self.assertFalse(content.get("thinking_enabled"))
        self.assertIsNotNone(content["response"])
        print("\ndeepseek-v3 Response:", content["response"])

    def test_deepseek_r1(self):
        """Test deepseek-r1 (thinking enabled, default model type)."""
        chat = DeepSeekChat(self.ds_session_id, self.authorization_token)
        res = chat.send_message("Say hello", printing=False, thinking_enabled=True, model_type="default")
        
        self.assertTrue(res.get("ok"), f"Model deepseek-r1 failed: {res}")
        self.assertIn("content", res)
        content = res["content"]
        self.assertIn("response", content)
        self.assertTrue(content.get("thinking_enabled"))
        self.assertIn("thought", content)
        self.assertIsNotNone(content["response"])
        self.assertIsNotNone(content["thought"])
        print("\ndeepseek-r1 Thought:", content["thought"])
        print("deepseek-r1 Response:", content["response"])

    def test_deepseek_v4(self):
        """Test deepseek-v4 (thinking disabled, expert model type)."""
        chat = DeepSeekChat(self.ds_session_id, self.authorization_token)
        res = chat.send_message("Say hello", printing=False, thinking_enabled=False, model_type="expert")
        
        self.assertTrue(res.get("ok"), f"Model deepseek-v4 failed: {res}")
        self.assertIn("content", res)
        content = res["content"]
        self.assertIn("response", content)
        self.assertFalse(content.get("thinking_enabled"))
        self.assertIsNotNone(content["response"])
        print("\ndeepseek-v4 Response:", content["response"])

    def test_deepseek_r4(self):
        """Test deepseek-r4 (thinking enabled, expert model type)."""
        chat = DeepSeekChat(self.ds_session_id, self.authorization_token)
        res = chat.send_message("Say hello", printing=False, thinking_enabled=True, model_type="expert")
        
        self.assertTrue(res.get("ok"), f"Model deepseek-r4 failed: {res}")
        self.assertIn("content", res)
        content = res["content"]
        self.assertIn("response", content)
        self.assertTrue(content.get("thinking_enabled"))
        self.assertIn("thought", content)
        self.assertIsNotNone(content["response"])
        self.assertIsNotNone(content["thought"])
        print("\ndeepseek-r4 Thought:", content["thought"])
        print("deepseek-r4 Response:", content["response"])

    def tearDown(self):
        import time
        time.sleep(3)

if __name__ == "__main__":
    unittest.main()
