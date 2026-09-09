# =============================================================
#  Transkription_Notes_Pipeline - tests/test_qa_detect.py
#  Unit tests for notes.detect_qa_start and notes.detect_qa_start_llm:
#  automatic detection of where a webinar's audience Q&A session begins,
#  which the rest of the Q&A pipeline then drives off qa_start_time_sec.
#
#  Pure text analysis. No whisperx/torch/PIL, and no real LLM call: the
#  fallback's backend is faked, so this runs in a bare environment.
#
#  Run: C:\Python\Python311\python.exe -m pytest tests/test_qa_detect.py -v
# =============================================================

import types

import notes


# -----------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------

def seg(start, text, end=None):
    return {"start": float(start),
            "end": float(end if end is not None else start + 5.0),
            "text": text}


def recording(cue_at=None, cue_text="", duration=600.0, step=10.0,
              filler="presenting the method today"):
    """Filler segments every `step` seconds up to `duration`, with cue_text
    placed at `cue_at`. With the defaults the searched window (the final 40
    percent) starts at 360s, so a cue at 480s sits inside it with 120s to
    spare."""
    segs = []
    t = 0.0
    while t < duration:
        segs.append(seg(t, cue_text if cue_at is not None and abs(t - cue_at) < 1e-9 else filler))
        t += step
    segs[-1]["end"] = duration
    return segs


# -----------------------------------------------------------------
# _normalize_cue_text
# -----------------------------------------------------------------

class TestNormalizeCueText:
    def test_lowercases_and_pads(self):
        assert notes._normalize_cue_text("Hello World") == " hello world "

    def test_strips_punctuation(self):
        assert notes._normalize_cue_text("Now, the questions!") == " now the questions "

    def test_ampersand_becomes_and(self):
        assert "q and a" in notes._normalize_cue_text("Time for the Q&A")

    def test_collapses_whitespace(self):
        assert notes._normalize_cue_text("a   b\n\tc") == " a b c "

    def test_keeps_umlauts(self):
        assert "fragerunde" in notes._normalize_cue_text("Fragerunde")
        assert "über" in notes._normalize_cue_text("Über")

    def test_blank_and_none(self):
        assert notes._normalize_cue_text("").strip() == ""
        assert notes._normalize_cue_text(None).strip() == ""


# -----------------------------------------------------------------
# detect_qa_start: positives
# -----------------------------------------------------------------

class TestDetectQaStartPositives:
    def test_english_now_to_the_questions_and_answers(self):
        """The phrasing this feature was asked for."""
        segs = recording(cue_at=480.0, cue_text="So, now to the questions and answers.")
        hit = notes.detect_qa_start(segs)
        assert hit is not None
        assert hit[0] == 480.0

    def test_german_kommen_wir_zu_den_fragen(self):
        segs = recording(cue_at=480.0, cue_text="Gut, dann kommen wir zu den Fragen.",
                         filler="Wir stellen die Methode vor")
        assert notes.detect_qa_start(segs)[0] == 480.0

    def test_german_single_word_fragerunde(self):
        segs = recording(cue_at=480.0, cue_text="Fragerunde.",
                         filler="Wir stellen die Methode vor")
        assert notes.detect_qa_start(segs)[0] == 480.0

    def test_ampersand_spelling(self):
        segs = recording(cue_at=480.0, cue_text="And now the Q&A!")
        assert notes.detect_qa_start(segs)[0] == 480.0

    def test_case_and_trailing_punctuation_are_ignored(self):
        segs = recording(cue_at=480.0, cue_text="NOW TO THE QUESTIONS AND ANSWERS!!!")
        assert notes.detect_qa_start(segs)[0] == 480.0

    def test_leading_filler_words_are_tolerated(self):
        segs = recording(cue_at=480.0,
                         cue_text="So, ähm, ok, dann kommen wir jetzt zu den Fragen",
                         filler="Wir stellen die Methode vor")
        assert notes.detect_qa_start(segs)[0] == 480.0

    def test_returns_the_matched_cue_alongside_the_time(self):
        segs = recording(cue_at=480.0, cue_text="Now to the questions and answers.")
        start, cue = notes.detect_qa_start(segs)
        assert start == 480.0
        assert cue in notes.QA_CUE_STRONG

    def test_weak_cue_is_detected_when_no_strong_one_exists(self):
        segs = recording(cue_at=480.0, cue_text="Are there any questions from the audience?")
        start, cue = notes.detect_qa_start(segs)
        assert start == 480.0
        assert cue in notes.QA_CUE_WEAK


# -----------------------------------------------------------------
# detect_qa_start: negatives and boundaries
# -----------------------------------------------------------------

class TestDetectQaStartNegatives:
    def test_no_cue_returns_none(self):
        assert notes.detect_qa_start(recording()) is None

    def test_empty_segments_returns_none(self):
        assert notes.detect_qa_start([]) is None

    def test_single_segment_recording(self):
        assert notes.detect_qa_start([seg(0.0, "Now to the questions and answers.")]) is None

    def test_segments_without_a_text_key_do_not_raise(self):
        segs = [{"start": 0.0, "end": 600.0}, {"start": 480.0, "end": 600.0}]
        assert notes.detect_qa_start(segs) is None

    def test_zero_duration_returns_none(self):
        assert notes.detect_qa_start([{"start": 0.0, "end": 0.0, "text": "q and a"}]) is None

    def test_email_me_housekeeping_is_never_a_cue(self):
        """The primary false-positive guard. This phrasing is deliberately
        absent from both tiers, so it is rejected wherever it appears."""
        early = recording(cue_at=180.0, cue_text="If you have any questions, just email me.")
        late = recording(cue_at=480.0, cue_text="If you have any questions, just email me.")
        assert notes.detect_qa_start(early) is None
        assert notes.detect_qa_start(late) is None

    def test_next_question_is_not_a_cue(self):
        segs = recording(cue_at=480.0, cue_text="Next question, please.")
        assert notes.detect_qa_start(segs) is None

    def test_intro_housekeeping_outside_the_window_is_ignored(self):
        """A real cue phrase, but at 5 percent of the recording: the presenter
        announcing that questions come later, not the session starting."""
        segs = recording(cue_at=30.0, cue_text="We will have time for questions at the end.")
        assert notes.detect_qa_start(segs) is None

    def test_cue_buried_deep_in_a_segment_is_rejected(self):
        long_lead = " ".join(["word"] * 30) + " now to the questions and answers"
        segs = recording(cue_at=480.0, cue_text=long_lead)
        assert notes.detect_qa_start(segs) is None

    def test_the_same_cue_near_the_segment_start_is_accepted(self):
        """Boundary partner of the test above: identical wording, short lead."""
        short_lead = " ".join(["word"] * 5) + " now to the questions and answers"
        segs = recording(cue_at=480.0, cue_text=short_lead)
        assert notes.detect_qa_start(segs)[0] == 480.0

    def test_cue_too_close_to_the_end_is_rejected(self):
        """A closing "thanks for the questions" must not create a 20-second
        Q&A that drops frames and burns an LLM call on an empty card."""
        segs = recording(cue_at=580.0, cue_text="Thanks for all the questions and answers.")
        assert notes.detect_qa_start(segs) is None

    def test_exactly_min_remaining_is_accepted(self):
        segs = recording(cue_at=540.0, cue_text="Now to the questions and answers.")
        assert notes.detect_qa_start(segs, min_remaining_sec=60.0)[0] == 540.0

    def test_search_window_parameter_is_respected(self):
        segs = recording(cue_at=380.0, cue_text="Now to the questions and answers.")
        assert notes.detect_qa_start(segs, search_last_fraction=0.4)[0] == 380.0
        assert notes.detect_qa_start(segs, search_last_fraction=0.1) is None

    def test_max_cue_offset_words_parameter_is_respected(self):
        long_lead = " ".join(["word"] * 20) + " now to the questions and answers"
        segs = recording(cue_at=480.0, cue_text=long_lead)
        assert notes.detect_qa_start(segs) is None
        assert notes.detect_qa_start(segs, max_cue_offset_words=25)[0] == 480.0


# -----------------------------------------------------------------
# Tier tie-breaking
# -----------------------------------------------------------------

class TestCueTiers:
    def test_last_strong_cue_wins(self):
        """Presenters foreshadow the Q&A before starting it, so the later
        announcement is the real boundary."""
        segs = recording()
        segs[38]["text"] = "Later we will do questions and answers."   # 380s
        segs[48]["text"] = "Right, now to the questions and answers."  # 480s
        assert notes.detect_qa_start(segs)[0] == 480.0

    def test_first_weak_cue_wins(self):
        """A weak opener recurs between questions, so the earliest one marks
        where Q&A actually opened."""
        segs = recording()
        for i in (40, 44, 48):
            segs[i]["text"] = "Are there any questions from the audience?"
        assert notes.detect_qa_start(segs)[0] == 400.0

    def test_a_strong_cue_beats_an_earlier_weak_one(self):
        segs = recording()
        segs[40]["text"] = "Are there any questions so far?"            # 400s, weak
        segs[46]["text"] = "Now to the questions and answers."          # 460s, strong
        assert notes.detect_qa_start(segs)[0] == 460.0

    def test_a_strong_cue_beats_a_later_weak_one(self):
        segs = recording()
        segs[44]["text"] = "Now to the questions and answers."          # 440s, strong
        segs[50]["text"] = "Are there any questions left?"              # 500s, weak
        assert notes.detect_qa_start(segs)[0] == 440.0


# -----------------------------------------------------------------
# Clock convention
# -----------------------------------------------------------------

class TestClockConvention:
    def test_returns_the_segment_start_verbatim(self):
        """Segments reach this function already rescaled to real time by
        run_pipeline._rescale_segments, and qa_start_time_sec uses that same
        clock, so the value must come back untouched - not rounded, not
        rescaled again."""
        segs = recording()
        segs[48] = seg(483.7, "Now to the questions and answers.")
        start, _ = notes.detect_qa_start(segs)
        assert start == 483.7
        assert start == segs[48]["start"]


# -----------------------------------------------------------------
# detect_qa_start_llm  (fallback, backend faked)
# -----------------------------------------------------------------

class TestDetectQaStartLlm:
    def _args(self, **over):
        args = dict(llm_backend="ollama", ollama_base_url="http://localhost:11434",
                    ollama_notes_model="qwen3:14b", anthropic_api_key="",
                    claude_model="")
        args.update(over)
        return args

    def test_parses_a_plain_number(self, monkeypatch):
        monkeypatch.setattr(notes, "_chat_ollama", lambda *a, **kw: "480")
        segs = recording()
        assert notes.detect_qa_start_llm(segs, **self._args()) == (480.0, "LLM")

    def test_tolerates_surrounding_words(self, monkeypatch):
        monkeypatch.setattr(notes, "_chat_ollama", lambda *a, **kw: "about 480.0 seconds")
        assert notes.detect_qa_start_llm(recording(), **self._args())[0] == 480.0

    def test_none_answer_returns_none(self, monkeypatch):
        monkeypatch.setattr(notes, "_chat_ollama", lambda *a, **kw: "NONE")
        assert notes.detect_qa_start_llm(recording(), **self._args()) is None

    def test_unparseable_answer_returns_none(self, monkeypatch):
        monkeypatch.setattr(notes, "_chat_ollama", lambda *a, **kw: "no idea, sorry")
        assert notes.detect_qa_start_llm(recording(), **self._args()) is None

    def test_answer_outside_the_window_is_rejected(self, monkeypatch):
        """Guards against the model answering with a time from the part of
        the recording it was never shown."""
        monkeypatch.setattr(notes, "_chat_ollama", lambda *a, **kw: "30")
        assert notes.detect_qa_start_llm(recording(), **self._args()) is None

    def test_answer_too_close_to_the_end_is_rejected(self, monkeypatch):
        monkeypatch.setattr(notes, "_chat_ollama", lambda *a, **kw: "590")
        assert notes.detect_qa_start_llm(recording(), **self._args()) is None

    def test_backend_exception_returns_none(self, monkeypatch):
        def boom(*a, **kw):
            raise RuntimeError("ollama is not running")
        monkeypatch.setattr(notes, "_chat_ollama", boom)
        assert notes.detect_qa_start_llm(recording(), **self._args()) is None

    def test_unknown_backend_returns_none_without_calling_anything(self, monkeypatch):
        def boom(*a, **kw):
            raise AssertionError("should not be called")
        monkeypatch.setattr(notes, "_chat_ollama", boom)
        assert notes.detect_qa_start_llm(recording(), **self._args(llm_backend="")) is None

    def test_anthropic_without_a_key_is_skipped(self):
        assert notes.detect_qa_start_llm(
            recording(), **self._args(llm_backend="anthropic",
                                      anthropic_api_key="sk-ant-...")) is None

    def test_anthropic_answer_is_parsed(self, monkeypatch):
        class FakeMessages:
            def create(self, **kwargs):
                block = types.SimpleNamespace(type="text", text="480")
                return types.SimpleNamespace(content=[block])

        class FakeClient:
            def __init__(self, api_key):
                self.messages = FakeMessages()

        monkeypatch.setitem(__import__("sys").modules, "anthropic",
                            types.SimpleNamespace(Anthropic=FakeClient))
        got = notes.detect_qa_start_llm(
            recording(), **self._args(llm_backend="anthropic",
                                      anthropic_api_key="sk-ant-real", claude_model="x"))
        assert got == (480.0, "LLM")

    def test_empty_segments_returns_none(self):
        assert notes.detect_qa_start_llm([], **self._args()) is None

    def test_only_the_tail_is_sent(self, monkeypatch):
        captured = {}

        def fake_chat(system, user, *a, **kw):
            captured["user"] = user
            return "NONE"

        monkeypatch.setattr(notes, "_chat_ollama", fake_chat)
        segs = recording()
        segs[0]["text"] = "OPENING REMARKS MARKER"
        segs[48]["text"] = "LATE SEGMENT MARKER"
        notes.detect_qa_start_llm(segs, **self._args())
        assert "LATE SEGMENT MARKER" in captured["user"]
        assert "OPENING REMARKS MARKER" not in captured["user"]
