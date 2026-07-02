# =============================================================
#  Transkription_Notes_Pipeline - keys_loader.py
#  Reads keys.cfg (or keys.cfg.example as fallback) into a dict.
#  Called once at startup from config.py.
#
#  File format: flat KEY=VALUE, # comments, blank lines ignored.
#  keys.cfg is excluded from Git; keys.cfg.example is the
#  committed safe template.
#
#  save_keys() / current_values() support the GUI Settings tab:
#  editing keys.cfg as a structured form instead of raw text.
# =============================================================

import logging
from pathlib import Path

log = logging.getLogger(__name__)

_HERE = Path(__file__).parent.resolve()
_KEYS_FILE    = _HERE / "keys.cfg"
_KEYS_EXAMPLE = _HERE / "keys.cfg.example"

# Keys the Settings form knows how to edit, in display order.
KNOWN_KEYS = [
    "LLM_BACKEND",
    "OLLAMA_BASE_URL",
    "OLLAMA_MODEL",
    "OLLAMA_VLM_MODEL",
    "ANTHROPIC_API_KEY",
    "HF_TOKEN",
]

# Keys whose values should be masked in the GUI (API keys / tokens).
SECRET_KEYS = {"ANTHROPIC_API_KEY", "HF_TOKEN"}


def _parse(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        result[key.strip()] = value.strip()
    return result


def load_keys() -> dict[str, str]:
    """
    Load secrets from keys.cfg.
    Falls back to keys.cfg.example with a warning if keys.cfg is missing.
    Returns an empty dict if neither file exists.
    """
    if _KEYS_FILE.exists():
        keys = _parse(_KEYS_FILE)
        log.debug("Loaded %d keys from %s", len(keys), _KEYS_FILE.name)
        return keys

    log.warning(
        "keys.cfg not found. "
        "Copy keys.cfg.example to keys.cfg and fill in your values. "
        "Falling back to keys.cfg.example (placeholder values only)."
    )
    if _KEYS_EXAMPLE.exists():
        return _parse(_KEYS_EXAMPLE)

    log.warning("Neither keys.cfg nor keys.cfg.example found. No secrets loaded.")
    return {}


def current_values() -> dict[str, str]:
    """Values to pre-fill the Settings form with: keys.cfg if it exists,
    otherwise the placeholder values from keys.cfg.example."""
    return load_keys()


def save_keys(values: dict[str, str]) -> Path:
    """
    Write keys.cfg from the annotated keys.cfg.example template, substituting
    the given values for matching keys and leaving every comment and blank
    line untouched. Keys not present in the template are appended at the end.

    Note: keys.cfg is only read once, at process startup (config.py imports
    keys_loader at module load time). Changes saved here take effect on the
    next run of the pipeline, not the current GUI session.
    """
    template = _KEYS_EXAMPLE.read_text(encoding="utf-8") if _KEYS_EXAMPLE.exists() else ""
    out_lines = []
    written = set()

    for line in template.splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#") and "=" in stripped:
            key = stripped.split("=", 1)[0].strip()
            if key in values:
                out_lines.append(f"{key}={values[key]}")
                written.add(key)
                continue
        out_lines.append(line)

    for key, value in values.items():
        if key not in written:
            out_lines.append(f"{key}={value}")

    _KEYS_FILE.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    log.info("keys.cfg saved: %s", _KEYS_FILE)
    return _KEYS_FILE
