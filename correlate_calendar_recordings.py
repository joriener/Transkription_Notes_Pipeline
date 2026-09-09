# =============================================================
#  Transkription_Notes_Pipeline - correlate_calendar_recordings.py
#  (task #94)
#
#  Correlates a calendar export (.ics, all VEVENTs) against a folder of
#  meeting recordings, matching each recording to the calendar event it
#  most likely belongs to. Built for the common case where a recording's
#  own filename/timestamp only tells you when it ENDED (e.g. a screen
#  recorder that names/saves the file at stop-time), not when it
#  started - so matching is done against each event's DTEND, not DTSTART.
#
#  Usage:
#    python correlate_calendar_recordings.py "X:\Agilent\Meetings\2026"
#    python correlate_calendar_recordings.py "X:\Agilent\Meetings\2026" --recursive
#    python correlate_calendar_recordings.py "X:\Agilent\Meetings\2026" ^
#        --ics "X:\Agilent\Meetings\2026\calendar.ics" --tolerance-min 30 ^
#        --output "X:\Agilent\Meetings\2026\correlation.csv"
#
#  If --ics is omitted, exactly one *.ics file must exist directly in
#  the given folder (not searched recursively even with --recursive,
#  since a calendar export is normally one file at the top level) - it
#  is used automatically. Otherwise, pass --ics explicitly.
#
#  A recording's own "end time" is taken from its filename if one is
#  recognizable there (same YYYYMMDD[_-]HHMMSS pattern used throughout
#  this project - see run_pipeline.extract_datetime_from_filename),
#  falling back to the file's last-modified time (its most common real-
#  world proxy for "when this recording finished/was saved") when the
#  filename has no time component.
#
#  Output: one CSV row per recording (see write_report_csv for columns),
#  matched against the calendar event whose end time is closest, within
#  --tolerance-min minutes. Recordings with no event that close are
#  still listed, with status "no_match_within_tolerance", so nothing
#  silently disappears from the report - you decide from there whether
#  to name that one manually.
#
#  A second CSV (task #96, write_batch_csv) is written alongside the
#  main report by default: "<output>_batch.csv", in the exact
#  file/language/title/date/comments format the GUI's Batch tab already
#  recognizes via its "Load .txt/.csv..." button, so every recording
#  loads straight into the batch queue with its matched Title/Date/
#  Comments pre-filled. Pass --no-batch-csv to skip it, or --batch-csv
#  to choose its path.
# =============================================================

import argparse
import csv
import logging
import os
from datetime import datetime
from pathlib import Path

import config
import ics_utils
import run_pipeline

log = logging.getLogger(__name__)

DEFAULT_TOLERANCE_MIN = 45.0


def recording_end_time(path: Path) -> tuple[datetime, str]:
    """
    Returns (datetime, source) where source is "filename" or "mtime".
    Prefers a full date+time parsed out of the filename (see
    run_pipeline.extract_datetime_from_filename); falls back to the
    file's own last-modified time when the filename has no recognizable
    time component (a date-only or generically-named file).
    """
    dt = run_pipeline.extract_datetime_from_filename(path.name)
    if dt is not None:
        return dt, "filename"
    return datetime.fromtimestamp(path.stat().st_mtime), "mtime"


def find_recordings(folder: Path, recursive: bool = False) -> list[Path]:
    """Every file directly in (or, with recursive=True, under) folder
    whose extension is one this project recognizes as audio/video input
    (config.SUPPORTED_EXTENSIONS - the same list the GUI file picker and
    --batch-folder discovery use), sorted by name for a stable, readable
    report order."""
    pattern_fn = folder.rglob if recursive else folder.glob
    return sorted(
        p for p in pattern_fn("*")
        if p.is_file() and p.suffix.lower() in config.SUPPORTED_EXTENSIONS
    )


def find_ics_file(folder: Path) -> Path | None:
    """Auto-discovery for --ics: only used when exactly one *.ics file
    sits directly in folder (never recursive - a calendar export is
    normally a single top-level file, and guessing among several would
    be more likely to pick the wrong one than to help)."""
    matches = sorted(folder.glob("*.ics"))
    return matches[0] if len(matches) == 1 else None


def correlate(events: list[dict], recordings: list[Path],
              tolerance_min: float = DEFAULT_TOLERANCE_MIN) -> list[dict]:
    """
    For each recording, find the calendar event whose "end" is closest
    to that recording's own end-timestamp (recording_end_time), and
    accept it as a match only if within tolerance_min minutes. Events
    with no usable "end" (DTSTART itself was missing/unparseable) are
    never picked as a match.

    Returns one row dict per recording, in the same order as
    `recordings`:
      {"recording": Path, "recording_end": datetime, "time_source": str,
       "matched_title": str | None, "event_start": datetime | None,
       "event_end": datetime | None, "diff_minutes": float | None,
       "status": "matched" | "no_match_within_tolerance" | "no_calendar_events",
       "attendees": list[dict], "agenda": str, "description": str}
    attendees/agenda/description (task #95) come from the matched
    event - "matched_title" already covers the event's Topic. They are
    [] / "" / "" for an unmatched row, since there is then no single
    event those fields could honestly be attributed to.
    """
    usable_events = [e for e in events if e.get("end") is not None]
    rows: list[dict] = []

    for rec in recordings:
        rec_end, source = recording_end_time(rec)

        if not usable_events:
            rows.append({
                "recording": rec, "recording_end": rec_end, "time_source": source,
                "matched_title": None, "event_start": None, "event_end": None,
                "diff_minutes": None, "status": "no_calendar_events",
                "attendees": [], "agenda": "", "description": "",
            })
            continue

        best = min(usable_events, key=lambda e: abs((e["end"] - rec_end).total_seconds()))
        diff_minutes = abs((best["end"] - rec_end).total_seconds()) / 60.0

        if diff_minutes <= tolerance_min:
            rows.append({
                "recording": rec, "recording_end": rec_end, "time_source": source,
                "matched_title": best["title"], "event_start": best["start"],
                "event_end": best["end"], "diff_minutes": round(diff_minutes, 1),
                "status": "matched",
                "attendees": best.get("attendees", []),
                "agenda": best.get("agenda", ""),
                "description": best.get("comments", ""),
            })
        else:
            rows.append({
                "recording": rec, "recording_end": rec_end, "time_source": source,
                "matched_title": None, "event_start": None, "event_end": None,
                "diff_minutes": round(diff_minutes, 1), "status": "no_match_within_tolerance",
                "attendees": [], "agenda": "", "description": "",
            })

    return rows


def write_report_csv(rows: list[dict], out_path: Path) -> None:
    """Writes the correlation report as CSV (utf-8-sig, so Excel opens
    it directly without mangling umlauts). One row per recording; see
    correlate()'s docstring for the row dict shape. Attendees (task
    #95) are rendered as one semicolon-separated 'Name <email>' string
    via ics_utils.format_attendees; Agenda/Description are blank for an
    unmatched row."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Recording", "Recording End Time", "Time Source", "Matched Meeting",
            "Meeting Start", "Meeting End", "Diff (min)", "Status",
            "Attendees", "Agenda", "Description",
        ])
        for r in rows:
            writer.writerow([
                r["recording"].name,
                r["recording_end"].strftime("%Y-%m-%d %H:%M:%S"),
                r["time_source"],
                r["matched_title"] or "",
                r["event_start"].strftime("%Y-%m-%d %H:%M:%S") if r["event_start"] else "",
                r["event_end"].strftime("%Y-%m-%d %H:%M:%S") if r["event_end"] else "",
                r["diff_minutes"] if r["diff_minutes"] is not None else "",
                r["status"],
                ics_utils.format_attendees(r.get("attendees", [])),
                r.get("agenda", ""),
                r.get("description", ""),
            ])


def write_batch_csv(rows: list[dict], out_path: Path) -> None:
    """
    Writes a second CSV, alongside the human-readable report, in the
    exact format the GUI's Batch tab already knows how to import
    unchanged via its "Load .txt/.csv..." button (gui_logic.
    csv_header_index matches "file"/"language"/"title"/"date"/
    "comments" case-insensitively against the Treeview's own columns -
    see gui.py's RunTabController._batch_load_path). Task #96.

    Column mapping from a correlate() row:
      file     - the recording's FULL path, OS-normalized via
                 os.path.normpath (task #97: on Windows this guarantees
                 backslashes even if r["recording"] was ever built from a
                 forward-slash string, e.g. one returned by a tkinter file
                 dialog) - not just the name, the GUI needs a path it can
                 open.
      language - always blank; the batch table's own per-tab language
                 default applies unless the user edits this cell.
      title    - the matched event's Topic (matched_title), blank if
                 unmatched.
      date     - the matched event's start date only (YYYY-MM-DD),
                 blank if unmatched (the batch table's "date" field is
                 a plain date, not a date+time).
      comments - Description, Agenda, and Attendees folded into one
                 field (there is no separate batch-table column for
                 those), one labeled block per line, e.g.:
                   Discuss GC-MS quote.
                   Agenda: 1. Intro
                   Invitees: Jane Doe <jane@example.com>
                 Blank if unmatched.

    Every recording gets a row, matched or not (same "nothing silently
    disappears" rule as write_report_csv) - an unmatched one just has
    blank title/date/comments, so it still shows up in the batch table
    ready to run, with meeting info to be filled in (or left blank) by
    hand.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "language", "title", "date", "comments"])
        for r in rows:
            comments_parts = []
            if r.get("description"):
                comments_parts.append(r["description"])
            if r.get("agenda"):
                comments_parts.append(f"Agenda: {r['agenda']}")
            if r.get("attendees"):
                comments_parts.append(f"Invitees: {ics_utils.format_attendees(r['attendees'])}")
            writer.writerow([
                os.path.normpath(str(r["recording"])),
                "",
                r["matched_title"] or "",
                r["event_start"].strftime("%Y-%m-%d") if r["event_start"] else "",
                "\n".join(comments_parts),
            ])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Correlate a calendar .ics export with a folder of meeting recordings, "
                    "matching each recording's end-of-file timestamp against calendar event "
                    "end times.")
    parser.add_argument("folder", help="Folder containing the recordings (and the .ics file, "
                                       "unless --ics is given explicitly)")
    parser.add_argument("--ics", help="Path to the .ics calendar file. Auto-discovered in "
                                      "--folder if omitted and exactly one .ics file is found "
                                      "directly there.")
    parser.add_argument("--recursive", action="store_true",
                       help="Also scan subfolders of --folder for recordings")
    parser.add_argument("--tolerance-min", type=float, default=DEFAULT_TOLERANCE_MIN,
                       help=f"Max minutes between a recording's end-timestamp and an event's "
                            f"end time to still count as a match (default: "
                            f"{DEFAULT_TOLERANCE_MIN:g})")
    parser.add_argument("--output", help="CSV report path (default: "
                                        "<folder>/calendar_recording_correlation.csv)")
    parser.add_argument("--batch-csv", help="Batch-table-ready CSV path (default: "
                                            "<output>_batch.csv next to the main report). "
                                            "Import this via the GUI's Batch tab -> "
                                            "\"Load .txt/.csv...\" to queue every recording "
                                            "with its matched Title/Date/Comments pre-filled.")
    parser.add_argument("--no-batch-csv", action="store_true",
                       help="Skip writing the batch-table-ready CSV, only the main report.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    folder = Path(args.folder)
    if not folder.is_dir():
        log.error("Not a folder: %s", folder)
        return 1

    ics_path = Path(args.ics) if args.ics else find_ics_file(folder)
    if ics_path is None:
        log.error(
            "No --ics given, and auto-discovery found zero or more than one .ics file "
            "directly in %s. Pass --ics \"path\\to\\calendar.ics\" explicitly.", folder)
        return 1
    if not ics_path.exists():
        log.error("ICS file not found: %s", ics_path)
        return 1

    events = ics_utils.parse_ics_events(str(ics_path))
    if not events:
        log.warning("No VEVENT entries found in %s - every recording will be reported as "
                    "\"no_calendar_events\".", ics_path)

    recordings = find_recordings(folder, recursive=args.recursive)
    if not recordings:
        log.error("No recordings found in %s (recognized extensions: %s).",
                 folder, sorted(config.SUPPORTED_EXTENSIONS))
        return 1

    rows = correlate(events, recordings, tolerance_min=args.tolerance_min)

    out_path = Path(args.output) if args.output else folder / "calendar_recording_correlation.csv"
    write_report_csv(rows, out_path)

    matched = sum(1 for r in rows if r["status"] == "matched")
    unmatched = len(rows) - matched
    print(f"{len(events)} calendar event(s), {len(recordings)} recording(s): "
         f"{matched} matched within {args.tolerance_min:g} min, {unmatched} not matched.")
    print(f"Report written to: {out_path}")

    if not args.no_batch_csv:
        batch_path = Path(args.batch_csv) if args.batch_csv else \
            out_path.with_name(out_path.stem + "_batch.csv")
        write_batch_csv(rows, batch_path)
        print(f"Batch-ready CSV written to: {batch_path}")
        print("Import it in the GUI: Batch mode -> \"Load .txt/.csv...\"")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
