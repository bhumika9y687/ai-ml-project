import json
import os
import unittest
from io import BytesIO
from urllib.error import HTTPError
from unittest.mock import patch

from utils.chatbot import DEFAULT_MODEL, get_ai_response


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
    def test_default_model_is_listed_groq_model(self):
        self.assertEqual(DEFAULT_MODEL, "openai/gpt-oss-20b")

    @patch.dict(os.environ, {"GROQ_API_KEY": "test-api-key"})
    @patch("utils.chatbot.urlopen")
    def test_sends_conversation_to_groq_and_returns_model_response(self, urlopen):
        urlopen.return_value = FakeResponse(
            {"choices": [{"message": {"content": "A generated response."}}]}
        )

        response = get_ai_response(
            "Can you explain my result?",
            [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi"}],
            {"risk_score": 33.6, "risk_label": "Moderate"},
            model="test-model",
        )

        self.assertEqual(response, "A generated response.")
        request = urlopen.call_args.args[0]
        self.assertEqual(
            request.full_url,
            "https://api.groq.com/openai/v1/chat/completions",
        )
        self.assertEqual(
            request.get_header("Authorization"), "Bearer test-api-key"
        )
        self.assertEqual(
            request.get_header("User-agent"), "CardioVue-Health-Assistant/1.0"
        )
        body = json.loads(request.data.decode("utf-8"))
        self.assertEqual(body["model"], "test-model")
        self.assertIn("Score: 33.6", body["messages"][0]["content"])
        self.assertEqual(body["messages"][-1]["content"], "Can you explain my result?")

    @patch("utils.chatbot.urlopen")
    def test_emergency_message_does_not_call_model(self, urlopen):
        response = get_ai_response("I have severe chest pain")

        self.assertIn("emergency", response.lower())
        urlopen.assert_not_called()

    @patch.dict(os.environ, {}, clear=True)
    @patch("utils.chatbot.urlopen")
    def test_requires_groq_api_key(self, urlopen):
        with self.assertRaisesRegex(RuntimeError, "GROQ_API_KEY"):
            get_ai_response("What does cholesterol mean?")

        urlopen.assert_not_called()

    @patch.dict(os.environ, {"GROQ_API_KEY": "test-api-key"})
    @patch("utils.chatbot.urlopen")
    def test_includes_groq_403_reason(self, urlopen):
        urlopen.side_effect = HTTPError(
            "https://api.groq.com",
            403,
            "Forbidden",
            hdrs=None,
            fp=BytesIO(
                b'{"error":{"message":"Access denied for this API key."}}'
            ),
        )

        with self.assertRaisesRegex(
            RuntimeError, "Groq returned HTTP 403. Access denied for this API key."
        ):
            get_ai_response("What does cholesterol mean?")


if __name__ == "__main__":
    unittest.main()
