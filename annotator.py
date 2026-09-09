# Transkription_Notes_Pipeline - annotator.py
# Sends slide snapshots to a local Ollama vision model for semantic annotation.
# Video mode only. Requires Ollama running at localhost:11434 with a
# vision-capable model loaded. Tested with: qwen2.5vl:7b (recommended),
# llava:13b, llava:7b, llama3.2-vision

import base64
import json
import logging
import time
from pathlib import Path

import requests

log = logging.getLogger(__name__)

DEFAULT_PROMPT = (
    "You are analysing a presentation slide screenshot. "
    "Return a JSON object with these keys: "
    "title (string), bullets (list of strings, max 5), "
    "slide_type (title|content|diagram|table|blank). "
    "Be concise. Return only valid JSON."
)


def build_prompt(base_prompt: str, output_language: str = "auto") -> str:
    """
    Append a language instruction to the VLM prompt so "title"/"bullets"
    are always written in a chosen language, independent of the slide's
    own language (e.g. a German slide deck annotated in English). Blank
    or "auto" (default) returns base_prompt unchanged: title/bullets
    follow the slide's own language, same as before this option existed.
    """
    if not output_language or output_language == "auto":
        return base_prompt
    from config import LANGUAGE_NAMES
    name = LANGUAGE_NAMES.get(output_language, output_language)
    return (
        base_prompt
        + f' Write "title" and "bullets" in {name}, regardless of the '
          "slide's own language. Keep proper nouns, product names, and "
          "abbreviations in their original form where translating them "
          "would be unnatural."
    )


def _encode_image(image_path: Path) -> str:
    """Base64-encode image for Ollama API payload."""
    with open(image_path, "rb") as fh:
        return base64.b64encode(fh.read()).decode("utf-8")


def annotate_slide(
    image_path: Path,
    model: str = "llava:13b",
    ollama_url: str = "http://localhost:11434/api/generate",
    prompt: str = DEFAULT_PROMPT,
    timeout_sec: int = 60,
    retries: int = 2,
) -> dict:
    """
    Call Ollama vision API for a single slide image.
    Returns a dict with keys: title, bullets, slide_type.
    Falls back to empty dict on failure.
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "images": [_encode_image(image_path)],
        "stream": False,
        "format": "json",
        "think": False,
        "options": {
            "num_predict": 512,    # Enough for title + 5 bullets; prevents mid-JSON truncation
            "temperature": 0.1,    # Low temperature for consistent structured output
            "num_ctx": 4096,       # Caps KV cache; without this Ollama defaults to the
                                    # model's full native context (262144 for qwen3-vl),
                                    # which overflows 16 GB VRAM and forces slow CPU offload
        },
    }

    for attempt in range(1, retries + 2):
        try:
            response = requests.post(
                ollama_url,
                json=payload,
                timeout=timeout_sec,
            )
            response.raise_for_status()
            data = response.json()
            # Thinking models (e.g. qwen3-vl) put the full output in "thinking"
            # and leave "response" empty, even with think=false.
            raw = data.get("response", "") or data.get("thinking", "")
            result = _safe_parse_json(raw)
            log.debug("Annotated %s: %s", image_path.name, result.get("title", "?"))
            return result
        except requests.exceptions.ConnectionError:
            log.error(
                "Ollama not reachable at %s. "
                "Ensure Ollama is running: `ollama serve`",
                ollama_url,
            )
            return {}
        except requests.exceptions.Timeout:
            log.warning("Ollama timeout (attempt %d/%d) for %s", attempt, retries + 1, image_path.name)
            if attempt <= retries:
                time.sleep(2 * attempt)
        except requests.exceptions.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 500:
                # Ollama 500 usually means the model ran out of memory on this image.
                # Retry with the image scaled down to reduce token count.
                log.warning(
                    "Ollama 500 on %s (attempt %d) — retrying with downscaled image.",
                    image_path.name, attempt,
                )
                if attempt <= retries:
                    try:
                        from PIL import Image as PILImage
                        with PILImage.open(image_path) as img:
                            img.thumbnail((512, 512))
                            import tempfile, pathlib
                            tmp = pathlib.Path(tempfile.mktemp(suffix=".jpg"))
                            img.save(tmp, format="JPEG", quality=75)
                            payload["images"] = [_encode_image(tmp)]
                            tmp.unlink(missing_ok=True)
                    except Exception as resize_exc:
                        log.debug("Could not downscale image: %s", resize_exc)
                    time.sleep(2)
            else:
                body = ""
                if exc.response is not None:
                    try:
                        body = exc.response.text[:500]
                    except Exception:
                        pass
                log.warning("HTTP error for %s: %s%s", image_path.name, exc,
                           f" | Ollama response: {body}" if body else "")
                return {}
        except Exception as exc:
            log.warning("Annotation failed for %s: %s", image_path.name, exc)
            return {}

    log.error("All annotation attempts failed for %s", image_path.name)
    return {}


def _safe_parse_json(text: str) -> dict:
    """
    Parse JSON from model output.
    Handles: markdown fences, truncated JSON (missing closing brackets),
    and partial bullet arrays cut off mid-string.
    """
    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]) if len(lines) > 2 else text
        text = text.strip()

    try:
        data = json.loads(text)
        return _extract_fields(data)
    except json.JSONDecodeError:
        pass

    repaired = _repair_json(text)
    if repaired:
        try:
            data = json.loads(repaired)
            log.debug("Repaired truncated VLM JSON response.")
            return _extract_fields(data)
        except json.JSONDecodeError:
            pass

    import re
    title_match = re.search(r'"title"\s*:\s*"([^"]+)"', text)
    title = title_match.group(1) if title_match else ""
    if title:
        log.debug("Extracted title via regex fallback: %s", title)
        return {"title": title, "bullets": [], "slide_type": "unknown"}

    log.warning("Could not parse VLM JSON response: %.300s", text)
    return {"title": "", "bullets": [], "slide_type": "unknown"}


def _flatten_bullet(item) -> str:
    """
    Coerce one bullet entry to a plain string. Weaker/quantized VLMs
    (observed with qwen2.5vl:3b) sometimes ignore the "list of strings"
    instruction and nest bullets as {"text": "..."} objects or as
    sub-lists (["a", "b"]) instead of flat strings. Normalising here
    keeps every downstream consumer (CSV, HTML report, PDF, DB) safe
    without needing its own defensive isinstance checks.
    """
    if isinstance(item, str):
        return item.strip()
    if isinstance(item, dict):
        for value in item.values():
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""
    if isinstance(item, list):
        parts = [_flatten_bullet(sub) for sub in item]
        return " ".join(p for p in parts if p)
    return str(item).strip()


def _extract_fields(data: dict) -> dict:
    bullets = data.get("bullets", [])
    if isinstance(bullets, str):
        bullets = [b.strip() for b in bullets.split(",") if b.strip()]
    elif isinstance(bullets, list):
        bullets = [_flatten_bullet(b) for b in bullets]
    else:
        bullets = []
    bullets = [b for b in bullets if b][:5]
    return {
        "title": str(data.get("title", "")),
        "bullets": bullets,
        "slide_type": str(data.get("slide_type", "content")),
    }


def _repair_json(text: str) -> str:
    """
    Attempt to close an unclosed JSON object by appending missing
    closing brackets and braces. Returns empty string if the input
    does not look like a JSON object at all.
    """
    text = text.strip()
    if not text.startswith("{"):
        return ""

    while text and text[-1] not in ('}', ']', '"', '0123456789'):
        last_comma = max(text.rfind(","), text.rfind(":"))
        if last_comma == -1:
            break
        text = text[:last_comma]
        break

    stack = []
    in_string = False
    escape = False
    for ch in text:
        if escape:
            escape = False
            continue
        if ch == '\\' and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch in '[{':
            stack.append(']' if ch == '[' else '}')
        elif ch in ']}':
            if stack and stack[-1] == ch:
                stack.pop()

    if in_string:
        text += '"'

    text += ''.join(reversed(stack))
    return text


def annotate_batch(
    slides: list,         # list of SlideChange objects
    snapshot_dir: Path,
    model: str = "llava:13b",
    ollama_url: str = "http://localhost:11434/api/generate",
    prompt: str = DEFAULT_PROMPT,
    timeout_sec: int = 60,
    stop_check=None,       # optional callable, checked between slides for a
                           # cooperative stop (see run_pipeline.py process_file);
                           # remaining slides are returned unannotated, not lost.
) -> list[dict]:
    """
    Annotate all slides and return enriched dicts.
    Skips VLM call if snapshot file does not exist.
    """
    results = []
    for slide in slides:
        if stop_check is not None and stop_check():
            log.info("Stop requested, VLM annotation halted at slide %d/%d.",
                     len(results) + 1, len(slides))
            for r in slides[len(results):]:
                snap = snapshot_dir / r.frame_path.name
                results.append({
                    "frame_index": r.frame_index, "timestamp_sec": r.timestamp_sec,
                    "snapshot_path": str(snap), "hash_value": r.hash_value,
                    "hamming_distance": r.hamming_distance,
                    "title": "", "bullets": [], "slide_type": "",
                })
            break
        # The staged snapshot keeps the extracted frame's own filename and
        # extension (run_pipeline copies it there unchanged), so no ".png"
        # assumption is needed here.
        snap = snapshot_dir / slide.frame_path.name
        annotation = {}
        if snap.exists():
            annotation = annotate_slide(
                snap, model=model, ollama_url=ollama_url,
                prompt=prompt, timeout_sec=timeout_sec,
            )
        results.append({
            "frame_index": slide.frame_index,
            "timestamp_sec": slide.timestamp_sec,
            "snapshot_path": str(snap),
            "hash_value": slide.hash_value,
            "hamming_distance": slide.hamming_distance,
            **annotation,
        })
    return results
