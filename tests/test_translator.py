# =============================================================
#  Transkription_Notes_Pipeline - tests/test_translator.py
#  Unit tests for translator.py: paragraph-aware chunking and the
#  translate_text() public entry point (both backends monkeypatched -
#  no live Ollama/Anthropic call is made).
# =============================================================

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import translator


class TestChunkText:
    def test_short_text_returned_as_single_chunk(self):
        text = "Hello world."
        assert translator._chunk_text(text, chunk_size=100) == [text]

    def test_splits_on_paragraph_boundaries(self):
        text = "Para one.\n\nPara two.\n\nPara three."
        chunks = translator._chunk_text(text, chunk_size=15)
        assert "".join(chunks).replace("\n\n", "") != ""
        # Every original paragraph must survive somewhere in the chunks.
        for para in ("Para one.", "Para two.", "Para three."):
            assert any(para in c for c in chunks)

    def test_oversized_single_paragraph_is_hard_split(self):
        # One paragraph, no blank lines, longer than chunk_size.
        text = "line one\n" * 50
        chunks = translator._chunk_text(text, chunk_size=40)
        assert len(chunks) > 1
        assert "".join(chunks).count("line one") == 50


class TestTranslateText:
    def test_blank_input_returned_unchanged(self):
        assert translator.translate_text(
            "   ", target_language="de", llm_backend="ollama",
            ollama_base_url="http://localhost:11434", ollama_translate_model="qwen3:14b",
            anthropic_api_key="", claude_model="",
        ) == "   "

    def test_ollama_backend_called_with_language_name_and_num_ctx(self, monkeypatch):
        captured = {}

        def fake_chat_ollama(system, user, ollama_base_url, model, timeout_sec,
                             num_predict=2048, num_ctx=8192):
            captured["system"] = system
            captured["user"] = user
            captured["num_ctx"] = num_ctx
            return "Hallo Welt"

        monkeypatch.setattr(translator, "_chat_ollama", fake_chat_ollama)
        result = translator.translate_text(
            "Hello world", target_language="de", llm_backend="ollama",
            ollama_base_url="http://localhost:11434", ollama_translate_model="qwen3:14b",
            anthropic_api_key="", claude_model="",
        )
        assert result == "Hallo Welt"
        assert "German" in captured["system"]
        assert captured["user"] == "Hello world"
        assert captured["num_ctx"] == 8_192

    def test_unknown_backend_returns_none(self):
        result = translator.translate_text(
            "Hello", target_language="de", llm_backend="bogus",
            ollama_base_url="", ollama_translate_model="", anthropic_api_key="", claude_model="",
        )
        assert result is None

    def test_chunk_failure_aborts_and_returns_none(self, monkeypatch):
        def failing_chat_ollama(*args, **kwargs):
            raise RuntimeError("connection refused")

        monkeypatch.setattr(translator, "_chat_ollama", failing_chat_ollama)
        result = translator.translate_text(
            "Hello world", target_language="de", llm_backend="ollama",
            ollama_base_url="http://localhost:11434", ollama_translate_model="qwen3:14b",
            anthropic_api_key="", claude_model="",
        )
        assert result is None

    def test_multi_chunk_results_joined_in_order(self, monkeypatch):
        calls = []

        def fake_chat_ollama(system, user, ollama_base_url, model, timeout_sec,
                             num_predict=2048, num_ctx=8192):
            calls.append(user)
            return f"[{user}]"

        monkeypatch.setattr(translator, "_chat_ollama", fake_chat_ollama)
        text = "Para one.\n\nPara two."
        result = translator.translate_text(
            text, target_language="de", llm_backend="ollama",
            ollama_base_url="http://localhost:11434", ollama_translate_model="qwen3:14b",
            anthropic_api_key="", claude_model="", chunk_size=10,
        )
        # chunk_size=10: each 9-char paragraph fits alone, but the pair
        # combined (20 chars incl. blank line) does not - one chunk each.
        assert len(calls) == 2
        assert result == "[Para one.]\n\n[Para two.]"
