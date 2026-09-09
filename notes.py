# =============================================================
#  Transkription_Notes_Pipeline - notes.py
#  LLM-generated meeting/webinar notes from a transcript.
#
#  Generalises the meeting_prompt.txt / webinar_prompt.txt toggle
#  from the two source pipelines into an open set of Markdown
#  prompt templates: any .md file in prompts/ is usable. The
#  template's first "## " heading is auto-detected and used both
#  to instruct the model and as assistant-prefill for Ollama.
#
#  For long transcripts (> single_pass_limit chars) a map-reduce
#  approach is used: split into chunks, summarise each chunk to
#  bullet notes, then merge into the final structured output.
#  This prevents hallucination caused by context-window overflow
#  on local models.
#
#  Backends:
#    ollama    - local LLM (OLLAMA_MODEL in keys.cfg)
#    anthropic - Claude API (ANTHROPIC_API_KEY in keys.cfg)
# =============================================================

import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt template discovery and loading
# ---------------------------------------------------------------------------

def list_prompt_templates(prompts_dir: Path) -> dict[str, Path]:
    """
    Return {display_name: path} for every .md file in prompts_dir,
    sorted alphabetically. Used to populate the GUI dropdown and to
    validate --prompt-template on the CLI.
    """
    prompts_dir = Path(prompts_dir)
    if not prompts_dir.is_dir():
        return {}
    templates = {}
    for path in sorted(prompts_dir.glob("*.md")):
        if path.stem.lower() == "readme":
            continue  # prompts/README.md documents the folder, it is not a template
        templates[path.stem] = path
    return templates


def load_prompt_template(path: Path) -> str:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8")


def first_heading(prompt_text: str) -> str:
    """Return the first '## ...' heading line found in the prompt template."""
    for line in prompt_text.splitlines():
        line = line.strip()
        if line.startswith("## "):
            return line
    return ""


def resolve_prompt_path(prompts_dir: Path, name_or_path: str) -> Path:
    """
    Resolve a --prompt-template value which may be a bare template name
    ("meeting"), a filename ("meeting.md") or a full/relative path.
    """
    candidate = Path(name_or_path)
    if candidate.is_file():
        return candidate
    prompts_dir = Path(prompts_dir)
    for suffix_try in (name_or_path, name_or_path + ".md"):
        p = prompts_dir / suffix_try
        if p.is_file():
            return p
    available = ", ".join(list_prompt_templates(prompts_dir).keys())
    raise FileNotFoundError(
        f"Prompt template '{name_or_path}' not found in {prompts_dir}. "
        f"Available: {available or '(none found)'}"
    )


# ---------------------------------------------------------------------------
# Transcript text helpers
# ---------------------------------------------------------------------------

def build_transcript_text_from_segments(segments: list[dict]) -> str:
    """Join whisperx segments into a readable timestamped transcript."""
    lines = []
    for seg in segments:
        spk = f"[{seg['speaker']}] " if seg.get("speaker") else ""
        m, s = divmod(int(seg.get("start", 0)), 60)
        h, m = divmod(m, 60)
        lines.append(f"[{h:02}:{m:02}:{s:02}] {spk}{seg.get('text', '').strip()}")
    return "\n".join(lines)


# Inserted into the transcript text at the Q&A boundary by
# build_transcript_text_with_qa_marker below - deliberately says WHAT it
# means and WHAT to do with it, so it stands on its own even for a model
# that only sees this one chunk (map-reduce) rather than the whole
# transcript with its surrounding explanation.
QA_SESSION_MARKER = (
    "=== Q&A SESSION BEGINS HERE - everything from this point on is the "
    "live audience Q&A segment; identify the real questions asked and the "
    "answers given in it for the QUESTIONS & ANSWERS section ==="
)


def build_transcript_text_with_qa_marker(
    segments: list[dict], qa_start_time_sec: float | None = None,
    qa_end_time_sec: float | None = None,
) -> str:
    """
    Like build_transcript_text_from_segments, but when qa_start_time_sec is
    set, inserts QA_SESSION_MARKER at that point in the transcript text.

    Task #61/#112: a single prompt/pass over the whole transcript (used
    whenever qa_include_in_summary is True - see run_pipeline.py's Step 3
    and run_notes_only) has to ask the model to recognise the Q&A portion
    from conversational cues alone, unless told otherwise - unreliable in
    practice (a wrap-up or a rambling discussion can read a lot like Q&A,
    and a real Q&A session can start without ever being announced as
    one). The pipeline already knows exactly where Q&A starts from
    qa_start_time_sec/qa_end_time_sec (set via the GUI's Q&A section or
    --qa-start/--qa-end); this marker is how that timing information -
    otherwise invisible to the model - gets passed to it, so the
    "## QUESTIONS & ANSWERS" section (see prompts/webinar.md) is filled in
    reliably instead of falling back to "No Q&A session" whenever the
    Q&A content did not.

    No-op (returns the plain, unmarked transcript) when qa_start_time_sec
    is not set, or no segment actually starts at/after it - the same
    "nothing configured, nothing changes" fallback the exclude path
    (qa_include_in_summary False) already uses.
    """
    if not qa_start_time_sec:
        return build_transcript_text_from_segments(segments)
    main_segments = [s for s in segments if s.get("start", 0) < qa_start_time_sec]
    qa_segments = [
        s for s in segments
        if s.get("start", 0) >= qa_start_time_sec
        and (not qa_end_time_sec or s.get("start", 0) < qa_end_time_sec)
    ]
    if not qa_segments:
        return build_transcript_text_from_segments(segments)
    main_text = build_transcript_text_from_segments(main_segments)
    qa_text = build_transcript_text_from_segments(qa_segments)
    return f"{main_text}\n\n{QA_SESSION_MARKER}\n\n{qa_text}"


# ---------------------------------------------------------------------------
# Q&A start detection (webinar recordings)
#
# The whole Q&A pipeline already runs off ONE number, qa_start_time_sec: the
# frame filter in process_file, the summary split, the synthetic qa_session
# slide, and generate_qa_pairs below all read it. Until now that number had
# to be typed in by hand (GUI "Q&A section" or --qa-start). These helpers
# work it out from the transcript instead.
#
# Cue phrases live here rather than in a module of their own because every
# other piece of Q&A semantics is already in this file (QA_SESSION_MARKER,
# build_transcript_text_with_qa_marker, _parse_qa_pairs, generate_qa_pairs),
# and this module imports only logging/re/pathlib, so it stays testable with
# no whisperx, torch or PIL anywhere near it.
#
# Two tiers, because the two kinds of phrase behave differently. A STRONG cue
# announces the session ("now to the questions and answers") and does not
# recur once it is under way, so the LAST one wins: presenters routinely
# foreshadow Q&A ("we'll take questions at the end") before actually starting
# it. A WEAK cue is an opener that recurs between questions ("are there any
# questions"), so the FIRST one wins, since that is when Q&A actually opened.
# ---------------------------------------------------------------------------

QA_CUE_STRONG = (
    # English
    "questions and answers", "question and answer", "q and a",
    "time for questions", "time for your questions",
    "take your questions", "take some questions",
    "open the floor", "open it up for questions",
    "move on to the questions", "over to the questions",
    "now to the questions", "question round",
    # German. Both the umlaut and the umlaut-less spelling are listed
    # because Whisper's own output varies between them.
    "fragen und antworten", "frage und antwort", "q und a",
    "fragerunde", "frageteil",
    "zu den fragen", "zu ihren fragen", "zu euren fragen",
    "zeit fur fragen", "zeit fuer fragen",
    "kommen wir zu den fragen",
    "fragen aus dem publikum", "fragen aus dem chat",
)

QA_CUE_WEAK = (
    # English
    "are there any questions", "do we have any questions",
    "any questions from", "first question",
    # German
    "gibt es fragen", "gibt es noch fragen",
    "haben sie fragen", "habt ihr fragen",
    "erste frage", "wer hat eine frage",
)

# Deliberately absent from both tiers, because these are mid-talk
# housekeeping or intra-Q&A transitions rather than a session boundary, and
# are the main source of false positives:
#   "if you have any questions", "any questions just email",
#   "next question", "naechste frage", "weitere fragen"

_CUE_NON_WORD = re.compile(r"[^\w]+", re.UNICODE)


def _normalize_cue_text(text: str) -> str:
    """
    Casefold, turn "&" into " and " so "Q&A" matches "q and a", replace every
    non-word character with a space (umlauts and sz survive, punctuation does
    not), collapse whitespace, and pad with single spaces so a plain
    substring test is word-boundary safe.
    """
    lowered = (text or "").casefold().replace("&", " and ")
    collapsed = _CUE_NON_WORD.sub(" ", lowered)
    return " " + " ".join(collapsed.split()) + " "


def _cue_word_offset(text_norm: str, phrase_norm: str) -> int | None:
    """How many words precede phrase_norm in text_norm, or None if absent."""
    idx = text_norm.find(phrase_norm.strip().join((" ", " ")))
    if idx < 0:
        return None
    return len(text_norm[:idx].split())


def detect_qa_start(
    segments: list[dict],
    search_last_fraction: float = 0.4,
    min_remaining_sec: float = 60.0,
    max_cue_offset_words: int = 12,
) -> tuple[float, str] | None:
    """
    Find where the audience Q&A session starts, from cue phrases in the
    transcript. Returns (start_seconds, matched_cue) or None.

    The returned value is on the SAME clock as the segments handed in.
    run_pipeline rescales segments to real time before this is called
    (_rescale_segments), and qa_start_time_sec lives on that same real-time
    clock, so the result must NOT be rescaled again by the caller.

    Pure: no logging, no config, no I/O, so the caller decides what to report.

    Four guards, all applied together, which between them rule out the
    everyday false positives:
      1. Only the final search_last_fraction of the recording is searched
         (default 0.4). A real Q&A block is the last 10-33 percent of a
         webinar, so 0.4 covers it with margin while excluding the intro
         housekeeping ("put your questions in the chat, we'll get to them at
         the end") that is the single most common false trigger.
      2. The cue must begin within the first max_cue_offset_words words of a
         segment (default 12), which rules out passing mentions buried
         mid-paragraph while still tolerating leading filler ("So, aehm, ok,
         dann kommen wir jetzt zu den Fragen").
      3. At least min_remaining_sec must remain afterwards (default 60), so a
         closing "thanks for all your questions" cannot create a 20-second
         Q&A that drops frames and burns an LLM call on an empty card.
      4. Tier tie-breaking: last STRONG cue, else first WEAK cue.
    """
    if not segments:
        return None

    ends = [s.get("end", s.get("start", 0.0)) or 0.0 for s in segments]
    duration = max(ends) if ends else 0.0
    if duration <= 0:
        return None

    window_start = duration * (1.0 - search_last_fraction)
    strong_hits: list[tuple[float, str]] = []
    weak_hits: list[tuple[float, str]] = []

    for seg in segments:
        start = seg.get("start", 0.0) or 0.0
        if start < window_start:
            continue
        if duration - start < min_remaining_sec:
            continue
        text_norm = _normalize_cue_text(seg.get("text", ""))
        if not text_norm.strip():
            continue
        for phrase in QA_CUE_STRONG:
            offset = _cue_word_offset(text_norm, phrase)
            if offset is not None and offset <= max_cue_offset_words:
                strong_hits.append((float(start), phrase))
                break
        else:
            for phrase in QA_CUE_WEAK:
                offset = _cue_word_offset(text_norm, phrase)
                if offset is not None and offset <= max_cue_offset_words:
                    weak_hits.append((float(start), phrase))
                    break

    if strong_hits:
        return strong_hits[-1]
    if weak_hits:
        return weak_hits[0]
    return None


_QA_LLM_SYSTEM = (
    "You are given the tail end of a webinar transcript, as timestamped lines. "
    "Decide where the live audience Q&A session begins: the point where the "
    "presenter stops presenting and starts taking questions from the audience. "
    "Answer with the start time in seconds of the first line that belongs to the "
    "Q&A session, as a bare number, nothing else. If there is no audience Q&A "
    "session in this text, answer exactly NONE."
)


def detect_qa_start_llm(
    segments: list[dict],
    llm_backend: str,
    ollama_base_url: str,
    ollama_notes_model: str,
    anthropic_api_key: str,
    claude_model: str,
    search_last_fraction: float = 0.4,
    min_remaining_sec: float = 60.0,
    ollama_num_ctx: int = 8_192,
    timeout_sec: int = 120,
) -> tuple[float, str] | None:
    """
    Fallback for when detect_qa_start finds no cue phrase: ask the configured
    notes backend where the Q&A starts. Returns (start_seconds, "LLM") or None.

    Only the TAIL of the transcript is sent, the same window detect_qa_start
    searches, not the whole thing. Returns None rather than raising for every
    failure mode (no backend configured, no credentials, unparseable answer,
    a time outside the window), so this can never turn a working
    transcription run into a failed one, and never makes Ollama or Anthropic
    a hard requirement of the pipeline.
    """
    if not segments:
        return None
    ends = [s.get("end", s.get("start", 0.0)) or 0.0 for s in segments]
    duration = max(ends) if ends else 0.0
    if duration <= 0:
        return None

    window_start = duration * (1.0 - search_last_fraction)
    tail = [s for s in segments if (s.get("start", 0.0) or 0.0) >= window_start]
    if not tail:
        return None

    lines = []
    for seg in tail:
        text = (seg.get("text", "") or "").strip()
        if text:
            lines.append(f"[{float(seg.get('start', 0.0) or 0.0):.1f}] {text}")
    if not lines:
        return None
    user = "\n".join(lines)

    raw = ""
    try:
        if llm_backend == "ollama":
            raw = _chat_ollama(_QA_LLM_SYSTEM, user, ollama_base_url,
                               ollama_notes_model, timeout_sec,
                               num_predict=16, num_ctx=ollama_num_ctx)
        elif llm_backend == "anthropic":
            if not anthropic_api_key or anthropic_api_key.startswith("sk-ant-..."):
                log.debug("Q&A auto-detect: no Anthropic key, skipping the LLM fallback.")
                return None
            import anthropic
            client = anthropic.Anthropic(api_key=anthropic_api_key)
            message = client.messages.create(
                model=claude_model, max_tokens=16,
                system=_QA_LLM_SYSTEM,
                messages=[{"role": "user", "content": user}],
            )
            for block in message.content:
                if getattr(block, "type", None) == "text":
                    raw = block.text
                    break
        else:
            log.debug("Q&A auto-detect: no usable llm_backend (%r), skipping the "
                      "LLM fallback.", llm_backend)
            return None
    except Exception as exc:
        log.debug("Q&A auto-detect: LLM fallback failed (%s) - no Q&A section set.", exc)
        return None

    answer = (raw or "").strip()
    if not answer or answer.upper().startswith("NONE"):
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", answer)
    if not match:
        log.debug("Q&A auto-detect: LLM answer %r is not a number.", answer[:60])
        return None
    value = float(match.group(0))
    if value < window_start or duration - value < min_remaining_sec:
        log.debug("Q&A auto-detect: LLM answered %.1fs, outside the searched "
                  "window (%.1f-%.1fs) - ignored.", value, window_start,
                  duration - min_remaining_sec)
        return None
    return value, "LLM"


# ---------------------------------------------------------------------------
# Chunking (map-reduce for long transcripts)
# ---------------------------------------------------------------------------

def _chunk_text(text: str, chunk_size: int) -> list[str]:
    """Split text into chunks at newline boundaries near chunk_size."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunks.append(text[start:])
            break
        split_at = text.rfind("\n", start, end)
        if split_at <= start:
            split_at = end
        chunks.append(text[start:split_at])
        start = split_at + 1
    return chunks


# ---------------------------------------------------------------------------
# Ollama backend (requests, /api/chat, assistant prefill)
# ---------------------------------------------------------------------------

def _chat_ollama(system: str, user: str, ollama_base_url: str, model: str,
                 timeout_sec: int, num_predict: int = 4096, prefill: str = "",
                 num_ctx: int = 16_384) -> str:
    """
    POST to Ollama /api/chat with optional assistant prefill.

    num_ctx is set explicitly (default 16384) rather than left to Ollama's
    own default. Without it, Ollama falls back to whatever the model's
    Modelfile declares - which varies wildly per model/quantization and
    is NOT guaranteed to be large enough to hold system prompt + full
    transcript + response. Too-small a context silently truncates the
    oldest messages (typically the transcript, since it is sent after
    the system prompt), which looks exactly like "the model ignored the
    transcript" / "notes come back empty or generic" without any error.
    16384 comfortably covers config.py's single_pass_limit (20000 chars)
    and chunk_size (12000 chars) plus response headroom for most models;
    lower it in config.py (see ollama_notes_num_ctx) if VRAM is tight.

    "think": False is set explicitly: hybrid reasoning models (Qwen3 and
    similar) can spend their entire response budget on the hidden
    reasoning phase and leave message.content empty, with the actual
    text only in message.thinking (observed 2026-07-18 with a
    qwen3-family model via translator.py's identical call - see that
    module's docstring for the full symptom). message.thinking is
    checked as a fallback below in case a given model ignores
    "think": False.
    """
    import requests
    url = ollama_base_url.rstrip("/") + "/api/chat"
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]
    if prefill:
        messages.append({"role": "assistant", "content": prefill})
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {"num_predict": num_predict, "temperature": 0.1, "num_ctx": num_ctx},
    }
    resp = requests.post(url, json=payload, timeout=timeout_sec)
    resp.raise_for_status()
    message = resp.json().get("message", {})
    content = (message.get("content", "") or message.get("thinking", "")).strip()
    return (prefill + content) if prefill else content


def language_instruction(output_language: str) -> str:
    """
    Build an instruction fragment forcing the notes LLM to write its
    response body in a specific language, independent of whatever
    language the transcript itself is in (e.g. an English call
    summarised into German notes). Blank or "auto" (the default)
    returns "" - no instruction added, current behaviour unchanged: the
    LLM responds in whichever language comes naturally, usually
    matching the transcript.

    Deliberately does not ask the model to translate section headings:
    those are enforced verbatim by _build_strict_system regardless of
    output_language, so instructing the model to also translate them
    would contradict that requirement.
    """
    if not output_language or output_language == "auto":
        return ""
    from config import LANGUAGE_NAMES
    name = LANGUAGE_NAMES.get(output_language, output_language)
    return (
        f"\n\nIMPORTANT: Write the body of your response in {name}, "
        "regardless of the language spoken/written in the transcript. "
        "Keep any section headings in their original template language. "
        "Keep proper nouns, product names, and technical abbreviations in "
        "their original form where translating them would be unnatural."
    )


def _build_strict_system(system_prompt: str, heading: str) -> str:
    """Add mandatory format enforcement to the system prompt."""
    if not heading:
        return system_prompt
    return (
        system_prompt
        + f"\n\nCRITICAL: Your response MUST begin with exactly '{heading}'."
        + "\nDo NOT write any introduction or preamble before that heading."
        + "\nOutput ALL sections in the exact order shown. No deviations."
    )


def _summarise_chunk_ollama(chunk: str, chunk_num: int, total: int,
                            ollama_base_url: str, model: str, timeout_sec: int,
                            num_ctx: int = 16_384) -> str:
    """Extract bullet-point notes from a single transcript chunk."""
    system = (
        "You are a precise note-taker. Extract key facts, topics, decisions, "
        "quotes, questions, answers, and action items from the transcript excerpt. "
        "Output only concise bullet points. Do not invent content. "
        f"This is part {chunk_num} of {total}."
    )
    user = f"---EXCERPT---\n{chunk}\n---END---"
    notes = _chat_ollama(system, user, ollama_base_url, model, timeout_sec,
                         num_predict=1024, num_ctx=num_ctx)
    log.info("Chunk %d/%d summarised (%d chars).", chunk_num, total, len(notes))
    return notes


def _ollama_generate(transcript_text: str, system_prompt: str, heading: str,
                     ollama_base_url: str, model: str, timeout_sec: int,
                     single_pass_limit: int, chunk_size: int,
                     num_ctx: int = 16_384) -> str | None:
    strict_system = _build_strict_system(system_prompt, heading)
    prefill = heading + "\n" if heading else ""

    if len(transcript_text) <= single_pass_limit:
        log.info("Transcript: %d chars - single-pass mode.", len(transcript_text))
        user_prompt = f"---TRANSCRIPT---\n{transcript_text}\n---END---"
        try:
            return _chat_ollama(strict_system, user_prompt, ollama_base_url, model,
                                timeout_sec, num_predict=4096, prefill=prefill, num_ctx=num_ctx)
        except Exception as e:
            log.error("Ollama error: %s", e)
            return None

    log.info("Transcript: %d chars > %d limit - map-reduce mode (chunk size %d).",
             len(transcript_text), single_pass_limit, chunk_size)
    chunks = _chunk_text(transcript_text, chunk_size)
    log.info("Map phase: %d chunks to summarise.", len(chunks))
    chunk_notes = []
    for i, chunk in enumerate(chunks, 1):
        try:
            chunk_notes.append(
                _summarise_chunk_ollama(chunk, i, len(chunks), ollama_base_url, model, timeout_sec,
                                        num_ctx=num_ctx)
            )
        except Exception as e:
            log.warning("Chunk %d failed: %s - skipping.", i, e)

    if not chunk_notes:
        log.error("All chunks failed - aborting notes generation.")
        return None

    combined = "\n\n".join(
        f"--- Notes from part {i+1} ---\n{n}" for i, n in enumerate(chunk_notes)
    )
    merge_user = (
        "The following are concise notes from different parts of a transcript. "
        "Consolidate into the structured format. Remove duplicates. "
        f"Begin your response with {heading}.\n\n"
        f"---NOTES---\n{combined}\n---END---"
    )
    log.info("Reduce phase: merging %d chunk summaries.", len(chunk_notes))
    try:
        return _chat_ollama(strict_system, merge_user, ollama_base_url, model,
                            timeout_sec, num_predict=4096, prefill=prefill, num_ctx=num_ctx)
    except Exception as e:
        log.error("Ollama merge error: %s", e)
        return None


# ---------------------------------------------------------------------------
# Anthropic backend
# ---------------------------------------------------------------------------

def _build_anthropic_request(transcript_text: str, system_prompt: str, heading: str,
                             claude_model: str) -> dict:
    """
    Build the Anthropic Messages API request body for a single notes-
    generation call, WITHOUT executing it. Shared by _anthropic_generate
    (synchronous, one call per file) and build_anthropic_batch_request
    (one entry in a Message Batches API submission, see run_pipeline.
    run_notes_batch_via_batch_api) - both need the exact same request
    shape, so it is built in exactly one place.

    Prompt caching (explicit breakpoint): the system prompt is the SAME
    text for every file processed with this recording_type/
    prompt_template/output_language in a run - only the user message
    (transcript + per-file header) varies. Marking it cache_control means
    every call after the first in a batch (--notes-batch, folder batch,
    several GUI runs within the cache TTL, or every request in an
    Anthropic Message Batch sharing this system prompt) reads this block
    from cache at ~10% of the normal input-token price instead of paying
    full price again. Below Anthropic's per-model minimum cacheable
    length (1024 tokens for the Sonnet family; see prompt-caching docs),
    the API silently processes the request without caching - no error,
    just no discount - which is possible for the shorter prompt
    templates in prompts/ (e.g. qa_summary.md), so this is a "free to
    try, sometimes a no-op" optimization, not a guaranteed saving on
    every template.
    """
    strict_system = _build_strict_system(system_prompt, heading)
    # Anthropic has a much larger context window; only truncate extreme cases.
    text = transcript_text
    if len(text) > 400_000:
        text = text[:400_000] + "\n\n[truncated]"
    user_prompt = f"---TRANSCRIPT---\n{text}\n---END---"
    return {
        "model": claude_model,
        # 8192 (was 4096): this budget is shared with any ThinkingBlock
        # the model emits before the actual answer (see _extract_text_block)
        # - 4096 left too little room for the notes text itself on longer
        # meetings once thinking ate into it.
        "max_tokens": 8192,
        "system": [{
            "type": "text",
            "text": strict_system,
            "cache_control": {"type": "ephemeral"},
        }],
        "messages": [{"role": "user", "content": user_prompt}],
    }


def _extract_text_block(content) -> str | None:
    """
    Return the first text block's text from an Anthropic Message's
    .content list, or None if there isn't one. Extended thinking (on by
    default for some models/accounts) prepends a ThinkingBlock
    (type="thinking") before the actual answer, so content[0] is not
    reliably the TextBlock - it has no .text attribute, which is what
    raised "'ThinkingBlock' object has no attribute 'text'" before this
    scan existed. Shared by the synchronous call (_anthropic_generate)
    and batch result processing (run_pipeline.run_notes_batch_via_batch_api),
    since both get back the same Message.content shape either way.
    """
    for block in content:
        if getattr(block, "type", None) == "text":
            return block.text
    return None


def _anthropic_generate(transcript_text: str, system_prompt: str, heading: str,
                        api_key: str, claude_model: str, timeout_sec: int) -> str | None:
    if not api_key or api_key.startswith("sk-ant-..."):
        log.error("ANTHROPIC_API_KEY not set in keys.cfg - skipping notes generation.")
        return None
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        request = _build_anthropic_request(transcript_text, system_prompt, heading, claude_model)
        message = client.messages.create(**request)
        text = _extract_text_block(message.content)
        if text is None:
            log.error("Anthropic response contained no text block (content: %r).", message.content)
        return text
    except Exception as e:
        log.error("Anthropic error: %s", e)
        return None


def _prepare_notes_inputs(
    transcript_text: str, prompt_path: Path, filename: str,
    meeting_title: str = "", meeting_date: str = "", meeting_comments: str = "",
    output_language: str = "auto",
) -> tuple[str, str, str]:
    """
    Load the prompt template and build (full_transcript_text, system_prompt,
    heading) - the three backend-agnostic inputs every notes-generation
    call needs, regardless of whether it runs synchronously
    (generate_notes) or as part of an Anthropic Message Batch
    (build_anthropic_batch_request, see run_pipeline.
    run_notes_batch_via_batch_api). Extracted out of generate_notes so
    both paths build this from the exact same logic - a batch submission
    must produce byte-identical system prompts to a synchronous run of
    the same file, or costs/output would silently diverge between the
    two modes.

    See generate_notes for what meeting_title/meeting_date/
    meeting_comments/output_language do.
    """
    from datetime import datetime

    system_prompt = load_prompt_template(prompt_path)
    heading = first_heading(system_prompt)
    if not heading:
        log.warning("No '## ' heading found in %s - format enforcement disabled.", prompt_path)

    system_prompt += language_instruction(output_language)

    header = f"Filename: {filename}\n"
    if meeting_title:
        header += f"Meeting title: {meeting_title}\n"
    header += f"Date: {meeting_date or datetime.now().strftime('%Y-%m-%d')}\n"
    if meeting_comments.strip():
        header += f"\nAdditional context/comments provided by the organizer:\n{meeting_comments.strip()}\n"
    header += "\n"
    full_transcript_text = header + "TRANSCRIPT:\n" + transcript_text

    return full_transcript_text, system_prompt, heading


def build_anthropic_batch_request(
    transcript_text: str, prompt_path: Path, filename: str, claude_model: str,
    meeting_title: str = "", meeting_date: str = "", meeting_comments: str = "",
    output_language: str = "auto",
) -> dict:
    """
    Build one Anthropic Messages API request body for this file, for
    submission as a single entry in an Anthropic Message Batch (see
    run_pipeline.run_notes_batch_via_batch_api and the Anthropic Batches
    API docs) instead of a live, synchronous _anthropic_generate call.
    Batch requests are billed at 50% of standard API prices and can
    stack with the prompt-caching discount on the shared system prompt
    (see _build_anthropic_request) - see run_notes_batch_via_batch_api
    for the submit/poll/retrieve flow this feeds into.

    Returns exactly the dict shape _build_anthropic_request would use
    for a live call with the same inputs - callers pass this straight
    into anthropic.types.messages.batch_create_params.Request's params.
    Does not touch the network at all; that happens once, for the whole
    batch, in run_notes_batch_via_batch_api.
    """
    full_transcript_text, system_prompt, heading = _prepare_notes_inputs(
        transcript_text, prompt_path, filename,
        meeting_title=meeting_title, meeting_date=meeting_date,
        meeting_comments=meeting_comments, output_language=output_language,
    )
    return _build_anthropic_request(full_transcript_text, system_prompt, heading, claude_model)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def generate_notes(
    transcript_text: str,
    prompt_path: Path,
    llm_backend: str,
    ollama_base_url: str,
    ollama_notes_model: str,
    anthropic_api_key: str,
    claude_model: str,
    filename: str,
    single_pass_limit: int = 20_000,
    chunk_size: int = 12_000,
    timeout_sec: int = 300,
    meeting_title: str = "",
    meeting_date: str = "",
    meeting_comments: str = "",
    output_language: str = "auto",
    ollama_num_ctx: int = 16_384,
) -> str | None:
    """
    Generate structured notes/summary from a transcript using the given
    Markdown prompt template. Returns the notes text (including the
    Filename/Date header) or None on failure.

    meeting_title/meeting_date (optional, set via the GUI's "Meeting info"
    fields, --meeting-title/--meeting-date, or --ics/"Load meeting info...")
    are added to the context header sent to the LLM so the summary can
    reference the actual meeting name/date instead of just the source
    filename. meeting_comments (task #93: same sources, plus a "Comments:"
    block in an imported .txt/.docx file) is included as its own labeled
    section in that same header, so organizer-provided context (agenda
    notes, background the transcript itself wouldn't mention, etc.)
    actually reaches the LLM and can inform the generated summary -
    rather than only ever appearing as a header line in the output
    documents (see reporter.py's meeting_comments usage).

    output_language ("auto" default, or a code like "de"/"en"/"fr"; see
    config.LANGUAGE_NAMES): forces the notes body into that language
    regardless of the transcript's own language. Applied to both the
    single-pass and the map-reduce merge phase (see language_instruction);
    the verbatim transcript itself is never touched by this.

    ollama_num_ctx (Ollama backend only, default 16384): explicit context
    window passed to Ollama, sized to comfortably hold single_pass_limit/
    chunk_size worth of transcript plus the response. Deliberately not
    left unset - Ollama's own per-model default varies (some models
    default far too small for a full transcript, silently truncating it;
    others default to their full native context, which can be far larger
    than needed and blow the available VRAM). Lower it in config.py
    (ollama_notes_num_ctx) if VRAM is tight.
    """
    full_transcript_text, system_prompt, heading = _prepare_notes_inputs(
        transcript_text, prompt_path, filename,
        meeting_title=meeting_title, meeting_date=meeting_date,
        meeting_comments=meeting_comments, output_language=output_language,
    )

    log.info("NOTES: backend='%s' template='%s'", llm_backend, Path(prompt_path).name)

    if llm_backend == "anthropic":
        body = _anthropic_generate(
            full_transcript_text, system_prompt, heading,
            anthropic_api_key, claude_model, timeout_sec,
        )
    elif llm_backend == "ollama":
        body = _ollama_generate(
            full_transcript_text, system_prompt, heading,
            ollama_base_url, ollama_notes_model, timeout_sec,
            single_pass_limit, chunk_size, num_ctx=ollama_num_ctx,
        )
    else:
        log.error("Unknown llm_backend '%s'. Use 'ollama' or 'anthropic'.", llm_backend)
        return None

    if not body:
        return None

    log.info("Notes generated (%d characters).", len(body))
    return body


# ---------------------------------------------------------------------------
# Q&A pair extraction for the slide report (separate Q&A slide, see
# run_pipeline._add_qa_slide)
# ---------------------------------------------------------------------------

def _parse_qa_pairs(raw: str) -> list[dict]:
    """
    Parse LLM output in the "- Q: ...\n  A: ..." bullet form (see
    generate_qa_pairs's system prompt) into [{"question":.., "answer":..}].
    Tolerant of minor formatting drift (missing leading "-", extra blank
    lines) since local Ollama models don't always follow a format
    instruction exactly. Returns [] if nothing matches - callers fall
    back to showing the raw transcript excerpt instead of an empty
    report card.
    """
    pairs = []
    matches = re.finditer(
        r"^[-*]?\s*Q:\s*(.+?)\s*\n\s*A:\s*(.+?)(?=\n\s*[-*]?\s*Q:|\Z)",
        raw.strip(), re.MULTILINE | re.DOTALL,
    )
    for m in matches:
        question = " ".join(m.group(1).split())
        answer = " ".join(m.group(2).split())
        if question or answer:
            pairs.append({"question": question, "answer": answer})
    return pairs


def generate_qa_pairs(
    qa_text: str,
    llm_backend: str,
    ollama_base_url: str,
    ollama_notes_model: str,
    anthropic_api_key: str,
    claude_model: str,
    output_language: str = "auto",
    ollama_num_ctx: int = 16_384,
    timeout_sec: int = 300,
) -> list[dict]:
    """
    Extract discrete audience Q&A exchanges from a raw Q&A-session
    transcript excerpt, for display as its own card in the slide report
    (see run_pipeline._add_qa_slide). Distinct from generate_notes()'s
    QUESTIONS & ANSWERS section: that one summarises the whole
    meeting/webinar through the recording_type's prompt template and only
    runs when notes/summary generation is requested; this one is a
    narrow, single-purpose extraction over ONLY the already-known Q&A
    time range (qa_start_time_sec/qa_end_time_sec), so it can run from
    Slide Review's "Rebuild outputs" too, which never calls
    generate_notes at all.

    Returns [] (never None) on empty input, any backend failure, or if
    the raw output can't be parsed into pairs - callers fall back to
    showing the raw transcript excerpt for that slide instead of an
    empty report card.
    """
    if not qa_text or not qa_text.strip():
        return []

    system = (
        "You are a precise transcript analyst. The following is the live "
        "audience Q&A portion of a webinar or meeting transcript. Identify "
        "every distinct question asked and the answer given to it. Do not "
        "invent content and do not include anything not clearly present in "
        "this excerpt. Consolidate a question that was restated or "
        "interrupted into a single exchange. Output ONLY a list of "
        "exchanges, one per question, in exactly this form and nothing "
        "else:\n"
        "- Q: <question>\n"
        "  A: <answer>\n"
        "If no question/answer exchange is present in this excerpt, output "
        "exactly: No Q&A identified."
    ) + language_instruction(output_language)

    text = qa_text
    max_chars = 400_000 if llm_backend == "anthropic" else 30_000
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[truncated]"
    user_prompt = f"---TRANSCRIPT---\n{text}\n---END---"

    try:
        if llm_backend == "anthropic":
            if not anthropic_api_key or anthropic_api_key.startswith("sk-ant-..."):
                log.warning("ANTHROPIC_API_KEY not set - skipping Q&A pair extraction for report.")
                return []
            import anthropic
            client = anthropic.Anthropic(api_key=anthropic_api_key)
            message = client.messages.create(
                model=claude_model, max_tokens=4096,
                system=[{"type": "text", "text": system}],
                messages=[{"role": "user", "content": user_prompt}],
            )
            raw = _extract_text_block(message.content) or ""
        elif llm_backend == "ollama":
            raw = _chat_ollama(system, user_prompt, ollama_base_url, ollama_notes_model,
                               timeout_sec, num_predict=2048, num_ctx=ollama_num_ctx)
        else:
            log.warning("Unknown llm_backend '%s' - skipping Q&A pair extraction for report.", llm_backend)
            return []
    except Exception as e:
        log.warning("Q&A pair extraction failed (%s) - falling back to raw transcript excerpt.", e)
        return []

    if not raw or raw.strip().lower().startswith("no q&a identified"):
        return []

    pairs = _parse_qa_pairs(raw)
    if not pairs:
        log.warning("Q&A pair extraction returned unparseable output - falling back to raw transcript excerpt.")
    else:
        log.info("Q&A pair extraction: %d exchange(s) identified.", len(pairs))
    return pairs
