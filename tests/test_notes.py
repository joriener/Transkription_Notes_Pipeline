# =============================================================
#  Transkription_Notes_Pipeline - tests/test_notes.py
#  Unit test for notes.generate_notes' context-header construction
#  (task #93): meeting_comments must reach the LLM as part of the
#  transcript context, not just end up in reporter.py's document
#  headers. The actual LLM call is monkeypatched out - this only
#  checks what text generate_notes builds and hands to the backend.
# =============================================================

import sys
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
                                  single_pass_limit, chunk_size):
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
