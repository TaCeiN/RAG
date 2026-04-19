import io
import json
import unittest

from app.adapters.ollama import OllamaGenerator


class _FakeResponse:
    def __init__(self, lines):
        self._lines = [line.encode("utf-8") for line in lines]
        self._index = 0

    def readline(self):
        if self._index >= len(self._lines):
            return b""
        line = self._lines[self._index]
        self._index += 1
        return line

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


class GeneratorOptionsTests(unittest.TestCase):
    def test_stream_chat_sends_sampling_options(self) -> None:
        captured = {}

        def opener(request):
            captured["url"] = request.full_url
            captured["body"] = json.loads(request.data.decode("utf-8"))
            return _FakeResponse(['{"message":{"content":"ok"},"done":false}\n', '{"done":true}\n'])

        gen = OllamaGenerator(
            "http://localhost:11434",
            "test-model",
            opener=opener,
            temperature=0.25,
            top_p=0.9,
            repeat_penalty=1.05,
        )

        events = list(gen.stream_chat([{"role": "user", "content": "x"}], think=False, keep_alive=-1))

        self.assertTrue(events)
        self.assertEqual(captured["url"], "http://localhost:11434/api/chat")
        self.assertIn("options", captured["body"])
        self.assertEqual(captured["body"]["options"]["temperature"], 0.25)
        self.assertEqual(captured["body"]["options"]["top_p"], 0.9)
        self.assertEqual(captured["body"]["options"]["repeat_penalty"], 1.05)


if __name__ == "__main__":
    unittest.main()
