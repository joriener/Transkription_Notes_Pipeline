# =============================================================
#  Transkription_Notes_Pipeline - requirements_check.py
#  Native Python dependency check, no PowerShell required. Backs
#  both install.ps1 -Verify (parity check) and the GUI Settings
#  tab "Check requirements" / "Install missing" buttons.
#
#  Usage:
#    python requirements_check.py            (print report, exit 1 if missing)
#    python requirements_check.py --install  (pip install missing packages)
# =============================================================

import importlib
import subprocess
import sys
from dataclasses import dataclass


@dataclass
class Requirement:
    module: str        # import name, e.g. "PIL"
    pip_name: str       # pip package name, e.g. "Pillow>=10.4.0"
    label: str           # display label
    required: bool        # False = optional / fallback, missing is not an error


# Kept in the same order as requirements.txt / install.ps1 -Verify for
# easy cross-reference when something drifts between the three.
REQUIREMENTS = [
    Requirement("PIL",        "Pillow>=10.4.0",      "Pillow",       True),
    Requirement("imagehash",  "ImageHash==4.3.2",    "ImageHash",    True),
    Requirement("numpy",      "numpy>=1.26.0",       "numpy",        True),
    Requirement("scipy",      "scipy>=1.12.0",       "scipy",        True),
    Requirement("pywt",       "PyWavelets>=1.6.0",   "PyWavelets",   True),
    Requirement("requests",   "requests>=2.32.0",    "requests",     True),
    Requirement("docx",       "python-docx>=1.1.0",  "python-docx",  True),
    Requirement("whisperx",   "whisperx==3.8.5",     "whisperx",     True),
    Requirement("torch",      "torch>=2.2.0",        "torch",        True),
    Requirement("torchaudio", "torchaudio>=2.2.0",   "torchaudio",   True),
    Requirement("tkinter",    "",                    "tkinter (GUI)", False),
    Requirement("playwright", "playwright>=1.45.0",  "Playwright (PDF, preferred)", False),
    Requirement("weasyprint", "weasyprint>=62.0",    "weasyprint (PDF fallback)",   False),
    Requirement("anthropic",  "anthropic>=0.40.0",   "anthropic (Claude backend)",  False),
]


def check_all() -> list[dict]:
    """Import-check every known dependency. Returns a list of dicts:
    {module, label, pip_name, required, installed}."""
    results = []
    for req in REQUIREMENTS:
        try:
            importlib.import_module(req.module)
            installed = True
        except Exception:
            installed = False
        results.append({
            "module": req.module, "label": req.label, "pip_name": req.pip_name,
            "required": req.required, "installed": installed,
        })
    return results


def missing_required(results: list[dict] | None = None) -> list[dict]:
    results = results if results is not None else check_all()
    return [r for r in results if r["required"] and not r["installed"]]


def missing_optional(results: list[dict] | None = None) -> list[dict]:
    results = results if results is not None else check_all()
    return [r for r in results if not r["required"] and not r["installed"]]


def format_report(results: list[dict] | None = None) -> str:
    results = results if results is not None else check_all()
    lines = []
    for r in results:
        mark = "OK  " if r["installed"] else "MISSING "
        tag = "" if r["required"] else " (optional)"
        lines.append(f"  [{mark}] {r['label']}{tag}")
    return "\n".join(lines)


def install_missing(results: list[dict] | None = None, python_exe: str | None = None,
                    include_optional: bool = True, log_fn=print) -> bool:
    """pip install every missing package that has a pip_name. Streams
    output through log_fn (defaults to print; the GUI passes its
    logging queue instead). Returns True if all installs succeeded."""
    results = results if results is not None else check_all()
    to_install = [r for r in results if not r["installed"] and r["pip_name"]
                  and (r["required"] or include_optional)]
    if not to_install:
        log_fn("Nothing to install: no missing packages with a known pip name.")
        return True

    python_exe = python_exe or sys.executable
    ok = True
    for r in to_install:
        log_fn(f"Installing {r['pip_name']} ...")
        try:
            proc = subprocess.run(
                [python_exe, "-m", "pip", "install", r["pip_name"]],
                capture_output=True, text=True, timeout=900,
            )
            for line in (proc.stdout or "").splitlines():
                log_fn(f"  {line}")
            if proc.returncode != 0:
                for line in (proc.stderr or "").splitlines():
                    log_fn(f"  {line}")
                log_fn(f"FAILED: {r['label']} (exit code {proc.returncode})")
                ok = False
            else:
                log_fn(f"OK: {r['label']} installed.")
        except Exception as exc:
            log_fn(f"FAILED: {r['label']} ({exc})")
            ok = False

    if any(r["module"] == "playwright" for r in to_install):
        log_fn("Playwright installed: downloading Chromium (one-time, ~300 MB) ...")
        try:
            proc = subprocess.run(
                [python_exe, "-m", "playwright", "install", "chromium"],
                capture_output=True, text=True, timeout=900,
            )
            for line in (proc.stdout or "").splitlines():
                log_fn(f"  {line}")
            if proc.returncode != 0:
                log_fn("Chromium download failed (non-fatal); PDF export falls back to weasyprint/pdfkit.")
        except Exception as exc:
            log_fn(f"Chromium download failed (non-fatal): {exc}")

    return ok


if __name__ == "__main__":
    results = check_all()
    print("Transkription_Notes_Pipeline - dependency check")
    print(format_report(results))
    missing_req = missing_required(results)
    if missing_req:
        print(f"\n{len(missing_req)} required package(s) missing.")
        if "--install" in sys.argv:
            install_missing(results)
        else:
            print("Run with --install to install them, or: pip install -r requirements.txt")
        sys.exit(1)
    print("\nAll required packages installed.")
    sys.exit(0)
