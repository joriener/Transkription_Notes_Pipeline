# =============================================================
#  Transkription_Notes_Pipeline - tests/test_notes.py
#  Unit test for notes.generate_notes' context-header construction
#  (task #93): meeting_comments must reach the LLM as part of the
#  transcript context, not just end up in reporter.py's document
#  headers. The actual LLM call is monkeypatched out - this only
#  checks what text generate_notes builds and hands to the backend.
# =============================================================

import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import notes


class TestGenerateNotesHeader:
    def _run(self, tmp_path, monkeypatch, **kwargs):
        prompt_path = tmp_path / "template.md"
        prompt_path.write_text("## Summary\nWrite a summary.\n", encoding="utf-8")

        captured = {}

        def fake_ollama_generate(full_transcript_text, system_prompt, heading,
                                  ollama_base_url, ollama_notes_model, timeout_sec,
                                  single_pass_limit, chunk_size, num_ctx=16_384):
            captured["full_transcript_text"] = full_transcript_text
            return "## Summary\nfake notes body"

        monkeypatch.setattr(notes, "_ollama_generate", fake_ollama_generate)

        notes.generate_notes(
            transcript_text="Speaker: hello world",
            prompt_path=prompt_path,
            llm_backend="ollama",
            ollama_base_url="http://localhost:11434",
            ollama_notes_model="qwen3:14b",
            anthropic_api_key="",
            claude_model="",
            filename="call.mp4",
            **kwargs,
        )
        return captured["full_transcript_text"]

    def test_comments_included_in_llm_context(self, tmp_path, monkeypatch):
        text = self._run(tmp_path, monkeypatch, meeting_comments="Discuss the Q3 budget.")
        assert "Discuss the Q3 budget." in text
        assert "Additional context/comments provided by the organizer:" in text

    def test_blank_comments_add_no_section(self, tmp_path, monkeypatch):
        text = self._run(tmp_path, monkeypatch, meeting_comments="")
        assert "Additional context/comments" not in text

    def test_comments_default_omitted_when_not_passed(self, tmp_path, monkeypatch):
        text = self._run(tmp_path, monkeypatch)
        assert "Additional context/comments" not in text

    def test_multiline_comments_preserved(self, tmp_path, monkeypatch):
        text = self._run(tmp_path, monkeypatch,
                          meeting_comments="Line one.\nLine two.")
        assert "Line one.\nLine two." in text

    def test_title_and_date_still_present_alongside_comments(self, tmp_path, monkeypatch):
        text = self._run(tmp_path, monkeypatch, meeting_title="Kickoff",
                          meeting_date="2026-07-04", meeting_comments="Bring laptop.")
        assert "Meeting title: Kickoff" in text
        assert "Date: 2026-07-04" in text
        assert "Bring laptop." in text


# ---------------------------------------------------------------------------
# _anthropic_generate: no prior test coverage existed for the Anthropic
# backend at all (only _ollama_generate was ever monkeypatched above). The
# real anthropic module is stubbed out via sys.modules so no live API call
# or installed SDK is required to run these.
# ---------------------------------------------------------------------------

class _FakeTextBlock:
    type = "text"

    def __init__(self, text):
        self.text = text


class _FakeThinkingBlock:
    type = "thinking"


class _FakeMessage:
    def __init__(self, content):
        self.content = content


def _install_fake_anthropic(monkeypatch, captured, content_blocks):
    class FakeMessages:
        def create(self, **kwargs):
            captured.update(kwargs)
            return _FakeMessage(content_blocks)

    class FakeClient:
        def __init__(self, api_key):
            captured["api_key"] = api_key
            self.messages = FakeMessages()

    fake_module = types.SimpleNamespace(Anthropic=FakeClient)
    monkeypatch.setitem(sys.modules, "anthropic", fake_module)
    return captured


class TestAnthropicGenerate:
    def test_missing_api_key_skips_call_returns_none(self, monkeypatch):
        captured = {}
        _install_fake_anthropic(monkeypatch, captured, [_FakeTextBlock("unused")])
        result = notes._anthropic_generate(
            "transcript", "system prompt", "## Summary",
            api_key="", claude_model="claude-sonnet-5", timeout_sec=60,
        )
        assert result is None
        assert "api_key" not in captured  # never even constructed the client

    def test_placeholder_api_key_skips_call_returns_none(self, monkeypatch):
        captured = {}
        _install_fake_anthropic(monkeypatch, captured, [_FakeTextBlock("unused")])
        result = notes._anthropic_generate(
            "transcript", "system prompt", "## Summary",
            api_key="sk-ant-...placeholder", claude_model="claude-sonnet-5", timeout_sec=60,
        )
        assert result is None
        assert "api_key" not in captured

    def test_system_prompt_sent_with_explicit_cache_control(self, monkeypatch):
        """Regression test: the system prompt must be the SAME text for
        every file processed with the same recording_type/prompt_template/
        output_language, so it is worth marking cache_control - a plain
        string system= would never be cacheable."""
        captured = {}
        _install_fake_anthropic(monkeypatch, captured, [_FakeTextBlock("## Summary\nfake body")])
        result = notes._anthropic_generate(
            "transcript text", "You are a note-taker.", "## Summary",
            api_key="sk-ant-real-key", claude_model="claude-sonnet-5", timeout_sec=60,
        )
        assert result == "## Summary\nfake body"
        system = captured["system"]
        assert isinstance(system, list)
        assert len(system) == 1
        assert system[0]["cache_control"] == {"type": "ephemeral"}
        assert "You are a note-taker." in system[0]["text"]
        assert captured["max_tokens"] == 8192

    def test_returns_first_text_block_skipping_thinking_block(self, monkeypatch):
        captured = {}
        _install_fake_anthropic(
            monkeypatch, captured,
            [_FakeThinkingBlock(), _FakeTextBlock("## Summary\nreal answer")])
        result = notes._anthropic_generate(
            "transcript", "system prompt", "## Summary",
            api_key="sk-ant-real-key", claude_model="claude-sonnet-5", timeout_sec=60,
        )
        assert result == "## Summary\nreal answer"

    def test_no_text_block_returns_none(self, monkeypatch):
        captured = {}
        _install_fake_anthropic(monkeypatch, captured, [_FakeThinkingBlock()])
        result = notes._anthropic_generate(
            "transcript", "system prompt", "## Summary",
            api_key="sk-ant-real-key", claude_model="claude-sonnet-5", timeout_sec=60,
        )
        assert result is None
