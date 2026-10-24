import json
import unittest
from unittest.mock import patch

from utils.chatbot import get_ai_response


class FakeResponse:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class ChatbotTests(unittest.TestCase):
    @patch("utils.chatbot.urlopen")
    def test_sends_conversation_to_local_ollama_and_returns_model_response(self, urlopen):
        urlopen.return_value = FakeResponse(
            {"message": {"content": "A generated response."}}
        )

        response = get_ai_response(
            "Can you explain my result?",
            [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}],
            {"risk_score": 33.6, "risk_label": "Moderate"},
            model="test-model",
        )

        self.assertEqual(response, "A generated response.")
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://127.0.0.1:11434/api/chat")
        body = json.loads(request.data.decode("utf-8"))
        self.assertEqual(body["model"], "test-model")
        self.assertFalse(body["stream"])
        self.assertIn("Score: 33.6", body["messages"][0]["content"])
        self.assertEqual(body["messages"][-1]["content"], "Can you explain my result?")

    @patch("utils.chatbot.urlopen")
    def test_emergency_message_does_not_call_model(self, urlopen):
        response = get_ai_response("I have severe chest pain")

        self.assertIn("emergency", response.lower())
        urlopen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
