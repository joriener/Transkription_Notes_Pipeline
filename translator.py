# =============================================================
#  Transkription_Notes_Pipeline - translator.py
#  LLM-based text translation for the "Translate" tab/CLI.
#
#  Mirrors notes.py's backend plumbing (ollama /api/chat + anthropic),
#  but the task here is pure translation, not summarisation: no heading
#  enforcement, no map-reduce merge phase. A long input is split into
#  chunks, each translated independently, and the results are joined
#  back in order - unlike notes.py, a translation task needs no final
#  consolidation LLM call (there is nothing to "merge", each chunk's
#  translation already stands on its own).
#
#  doc_translate.py is the caller: it reads text out of each supported
#  file format (txt/docx/pptx/pdf/image) and calls translate_text() per
#  paragraph/text unit. This module has no file-format knowledge at all,
#  and doc_translate.py has no LLM/network knowledge - kept separate the
#  same way notes.py (LLM) and reporter.py (file writers) are.
# =============================================================

import logging

log = logging.getLogger(__name__)

SYSTEM_PROMPT_TEMPLATE = (
    "You are a professional translator. Translate the following text into {language}. "
    "Preserve the original meaning, tone, register, and paragraph/line structure exactly. "
    "Do not add commentary, explanations, notes, or quotation marks around the output. "
    "Do not translate proper nouns, product names, code, or URLs unless they have an "
    "established translation in {language}. Output ONLY the translated text, nothing else."
)


# ---------------------------------------------------------------------------
# Chunking (paragraph-aware; no reduce phase needed for translation)
# ---------------------------------------------------------------------------

def _chunk_text(text: str, chunk_size: int) -> list[str]:
    """
    Split text into chunks at paragraph (blank-line) boundaries near
    chunk_size chars. Paragraph-aware rather than a plain character
    split (see notes._chunk_text) because preserving paragraph breaks
    matters more for document translation than for note extraction -
    most callers here (doc_translate.py) already pass one short
    paragraph at a time, so this mostly only engages for a large plain
    .txt file or a PDF/OCR page translated as one blob.
    """
    if len(text) <= chunk_size:
        return [text]

    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        candidate = (current + "\n\n" + para) if current else para
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(para) <= chunk_size:
            current = para
        else:
            # A single paragraph alone exceeds chunk_size (e.g. one huge
            # block with no blank lines) - hard-split at the nearest
            # newline as a last resort.
            start = 0
            while start < len(para):
                end = start + chunk_size
                if end >= len(para):
                    chunks.append(para[start:])
                    start = end
                    break
                split_at = para.rfind("\n", start, end)
                if split_at <= start:
                    split_at = end
                chunks.append(para[start:split_at])
                start = split_at
            current = ""
    if current:
        chunks.append(current)
    return chunks


# ---------------------------------------------------------------------------
# Ollama backend
# ---------------------------------------------------------------------------

def _chat_ollama(system: str, user: str, ollama_base_url: str, model: str,
                 timeout_sec: int, num_predict: int = 2048, num_ctx: int = 8_192) -> str:
    """
    POST to Ollama /api/chat. num_ctx is set explicitly (default 8192) -
    see notes._chat_ollama's docstring for why an unset num_ctx is
    risky (silent transcript/text truncation on some models). 8192 is
    smaller than notes.py's 16384 default since translation units are
    usually one paragraph at a time; raise ollama_translate_num_ctx in
    config.py if you translate very long single blocks (e.g. whole
    unstructured .txt files with no paragraph breaks).

    "think": False is set explicitly: hybrid reasoning models (Qwen3 and
    similar) can spend their entire response budget on the hidden
    reasoning phase and leave message.content empty, with the actual
    text only in message.thinking - observed 2026-07-18 with a
    qwen3-family Ollama model taking ~90s per call and then failing
    with "Empty translation result". Disabling thinking for a plain
    translation task (no reasoning needed) avoids the wasted time
    entirely; message.thinking is still checked as a fallback below in
    case a given model ignores "think": False.
    """
    import requests
    url = ollama_base_url.rstrip("/") + "/api/chat"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "think": False,
        "options": {"num_predict": num_predict, "temperature": 0.1, "num_ctx": num_ctx},
    }
    resp = requests.post(url, json=payload, timeout=timeout_sec)
    resp.raise_for_status()
    message = resp.json().get("message", {})
    return (message.get("content", "") or message.get("thinking", "")).strip()


# ---------------------------------------------------------------------------
# Anthropic backend
# ---------------------------------------------------------------------------

def _anthropic_translate(text: str, system: str, api_key: str, claude_model: str,
                         timeout_sec: int) -> str | None:
    if not api_key or api_key.startswith("sk-ant-..."):
        log.error("ANTHROPIC_API_KEY not set in keys.cfg - skipping translation.")
        return None
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=claude_model,
            max_tokens=8192,
            system=system,
            messages=[{"role": "user", "content": text}],
        )
        for block in message.content:
            if getattr(block, "type", None) == "text":
                return block.text
        log.error("Anthropic translation error: no text block in response (got %s).",
                  [getattr(b, "type", None) for b in message.content])
        return None
    except Exception as e:
        log.error("Anthropic translation error: %s", e)
        return None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def translate_text(
    text: str,
    target_language: str,
    llm_backend: str,
    ollama_base_url: str,
    ollama_translate_model: str,
    anthropic_api_key: str,
    claude_model: str,
    timeout_sec: int = 120,
    chunk_size: int = 6_000,
    ollama_num_ctx: int = 8_192,
) -> str | None:
    """
    Translate a block of text into target_language. Returns the
    translated text, the original text unchanged if it is blank/
    whitespace-only (nothing to translate), or None on failure.

    target_language must be a real language code from
    config.LANGUAGE_NAMES (e.g. "de", "en") - unlike notes.py's
    output_language, "auto" is not meaningful here: a translation run
    always has an explicit target, validated by the caller (GUI/CLI)
    before this is ever reached.
    """
    if not text or not text.strip():
        return text

    from config import LANGUAGE_NAMES
    language_name = LANGUAGE_NAMES.get(target_language, target_language) or target_language
    system = SYSTEM_PROMPT_TEMPLATE.format(language=language_name)

    chunks = _chunk_text(text, chunk_size)
    translated_chunks = []
    for i, chunk in enumerate(chunks, 1):
        try:
            if llm_backend == "anthropic":
                result = _anthropic_translate(chunk, system, anthropic_api_key, claude_model, timeout_sec)
            elif llm_backend == "ollama":
                result = _chat_ollama(system, chunk, ollama_base_url, ollama_translate_model,
                                      timeout_sec, num_ctx=ollama_num_ctx)
            else:
                log.error("Unknown llm_backend '%s'. Use 'ollama' or 'anthropic'.", llm_backend)
                return None
        except Exception as e:
            log.error("Translation error on chunk %d/%d: %s", i, len(chunks), e)
            return None
        if not result:
            log.error("Empty translation result for chunk %d/%d.", i, len(chunks))
            return None
        translated_chunks.append(result)

    return "\n\n".join(translated_chunks)
