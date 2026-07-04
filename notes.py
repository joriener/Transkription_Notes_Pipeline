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
                 timeout_sec: int, num_predict: int = 4096, prefill: str = "") -> str:
    """POST to Ollama /api/chat with optional assistant prefill."""
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
        "options": {"num_predict": num_predict, "temperature": 0.1},
    }
    resp = requests.post(url, json=payload, timeout=timeout_sec)
    resp.raise_for_status()
    content = resp.json().get("message", {}).get("content", "").strip()
    return (prefill + content) if prefill else content


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
                            ollama_base_url: str, model: str, timeout_sec: int) -> str:
    """Extract bullet-point notes from a single transcript chunk."""
    system = (
        "You are a precise note-taker. Extract key facts, topics, decisions, "
        "quotes, questions, answers, and action items from the transcript excerpt. "
        "Output only concise bullet points. Do not invent content. "
        f"This is part {chunk_num} of {total}."
    )
    user = f"---EXCERPT---\n{chunk}\n---END---"
    notes = _chat_ollama(system, user, ollama_base_url, model, timeout_sec, num_predict=1024)
    log.info("Chunk %d/%d summarised (%d chars).", chunk_num, total, len(notes))
    return notes


def _ollama_generate(transcript_text: str, system_prompt: str, heading: str,
                     ollama_base_url: str, model: str, timeout_sec: int,
                     single_pass_limit: int, chunk_size: int) -> str | None:
    strict_system = _build_strict_system(system_prompt, heading)
    prefill = heading + "\n" if heading else ""

    if len(transcript_text) <= single_pass_limit:
        log.info("Transcript: %d chars - single-pass mode.", len(transcript_text))
        user_prompt = f"---TRANSCRIPT---\n{transcript_text}\n---END---"
        try:
            return _chat_ollama(strict_system, user_prompt, ollama_base_url, model,
                                timeout_sec, num_predict=4096, prefill=prefill)
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
                _summarise_chunk_ollama(chunk, i, len(chunks), ollama_base_url, model, timeout_sec)
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
                            timeout_sec, num_predict=4096, prefill=prefill)
    except Exception as e:
        log.error("Ollama merge error: %s", e)
        return None


# ---------------------------------------------------------------------------
# Anthropic backend
# ---------------------------------------------------------------------------

def _anthropic_generate(transcript_text: str, system_prompt: str, heading: str,
                        api_key: str, claude_model: str, timeout_sec: int) -> str | None:
    if not api_key or api_key.startswith("sk-ant-..."):
        log.error("ANTHROPIC_API_KEY not set in keys.cfg - skipping notes generation.")
        return None
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        strict_system = _build_strict_system(system_prompt, heading)
        # Anthropic has a much larger context window; only truncate extreme cases.
        text = transcript_text
        if len(text) > 400_000:
            text = text[:400_000] + "\n\n[truncated]"
        user_prompt = f"---TRANSCRIPT---\n{text}\n---END---"
        message = client.messages.create(
            model=claude_model,
            max_tokens=4096,
            system=strict_system,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return message.content[0].text
    except Exception as e:
        log.error("Anthropic error: %s", e)
        return None


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
    """
    from datetime import datetime

    system_prompt = load_prompt_template(prompt_path)
    heading = first_heading(system_prompt)
    if not heading:
        log.warning("No '## ' heading found in %s - format enforcement disabled.", prompt_path)

    header = f"Filename: {filename}\n"
    if meeting_title:
        header += f"Meeting title: {meeting_title}\n"
    header += f"Date: {meeting_date or datetime.now().strftime('%Y-%m-%d')}\n"
    if meeting_comments.strip():
        header += f"\nAdditional context/comments provided by the organizer:\n{meeting_comments.strip()}\n"
    header += "\n"
    full_transcript_text = header + "TRANSCRIPT:\n" + transcript_text

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
            single_pass_limit, chunk_size,
        )
    else:
        log.error("Unknown llm_backend '%s'. Use 'ollama' or 'anthropic'.", llm_backend)
        return None

    if not body:
        return None

    log.info("Notes generated (%d characters).", len(body))
    return body
