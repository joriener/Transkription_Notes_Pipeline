# =============================================================
#  Transkription_Notes_Pipeline - tests/test_correlate_calendar_recordings.py
#  Unit tests for correlate_calendar_recordings.py (task #94): matching
#  a folder of meeting recordings against a calendar .ics export by
#  comparing each recording's own end-of-file timestamp to each event's
#  end time. No ffmpeg/whisperx/pyannote needed - this only touches
#  filenames, file mtimes, and .ics parsing.
#
#  Run: pytest tests/test_correlate_calendar_recordings.py -v
# =============================================================

import csv
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import correlate_calendar_recordings as ccr


def make_event(title, start, end=None, comments="", agenda="", attendees=None):
    return {"title": title, "start": start, "end": end if end is not None else start,
            "all_day": False, "comments": comments, "agenda": agenda,
            "attendees": attendees if attendees is not None else []}


# -----------------------------------------------------------------
# recording_end_time
# -----------------------------------------------------------------

class TestRecordingEndTime:
    def test_prefers_filename_timestamp(self, tmp_path):
        p = tmp_path / "Video_2026-07-04_154005.mp4"
        p.write_bytes(b"fake")
        dt, source = ccr.recording_end_time(p)
        assert dt == datetime(2026, 7, 4, 15, 40, 5)
        assert source == "filename"

    def test_falls_back_to_mtime_when_no_time_in_filename(self, tmp_path):
        p = tmp_path / "recording.mp4"
        p.write_bytes(b"fake")
        # Set a known mtime so the assertion is exact, not just "close".
        known = time.mktime(datetime(2026, 7, 4, 10, 0, 0).timetuple())
        os.utime(p, (known, known))
        dt, source = ccr.recording_end_time(p)
        assert dt == datetime(2026, 7, 4, 10, 0, 0)
        assert source == "mtime"

    def test_date_only_filename_falls_back_to_mtime(self, tmp_path):
        # A date with no HHMMSS component isn't precise enough to use -
        # extract_datetime_from_filename returns None for it, so this
        # must fall back too, not silently use midnight.
        p = tmp_path / "2026-07-04.wav"
        p.write_bytes(b"fake")
        known = time.mktime(datetime(2026, 7, 4, 11, 30, 0).timetuple())
        os.utime(p, (known, known))
        dt, source = ccr.recording_end_time(p)
        assert source == "mtime"
        assert dt == datetime(2026, 7, 4, 11, 30, 0)


# -----------------------------------------------------------------
# find_recordings
# -----------------------------------------------------------------

class TestFindRecordings:
    def test_only_recognized_extensions(self, tmp_path):
        (tmp_path / "a.mp4").write_bytes(b"x")
        (tmp_path / "b.wav").write_bytes(b"x")
        (tmp_path / "c.txt").write_bytes(b"x")
        (tmp_path / "d.json").write_bytes(b"x")
        found = ccr.find_recordings(tmp_path)
        assert {p.name for p in found} == {"a.mp4", "b.wav"}

    def test_non_recursive_by_default(self, tmp_path):
        (tmp_path / "top.mp4").write_bytes(b"x")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "nested.mp4").write_bytes(b"x")
        found = ccr.find_recordings(tmp_path, recursive=False)
        assert {p.name for p in found} == {"top.mp4"}

    def test_recursive_includes_subfolders(self, tmp_path):
        (tmp_path / "top.mp4").write_bytes(b"x")
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "nested.mp4").write_bytes(b"x")
        found = ccr.find_recordings(tmp_path, recursive=True)
        assert {p.name for p in found} == {"top.mp4", "nested.mp4"}

    def test_empty_folder_returns_empty_list(self, tmp_path):
        assert ccr.find_recordings(tmp_path) == []


# -----------------------------------------------------------------
# find_ics_file
# -----------------------------------------------------------------

class TestFindIcsFile:
    def test_single_ics_file_found(self, tmp_path):
        p = tmp_path / "calendar.ics"
        p.write_text("BEGIN:VCALENDAR\nEND:VCALENDAR\n", encoding="utf-8")
        assert ccr.find_ics_file(tmp_path) == p

    def test_no_ics_file_returns_none(self, tmp_path):
        assert ccr.find_ics_file(tmp_path) is None

    def test_multiple_ics_files_returns_none(self, tmp_path):
        (tmp_path / "a.ics").write_text("x", encoding="utf-8")
        (tmp_path / "b.ics").write_text("x", encoding="utf-8")
        assert ccr.find_ics_file(tmp_path) is None


# -----------------------------------------------------------------
# correlate
# -----------------------------------------------------------------

class TestCorrelate:
    def test_matches_closest_event_within_tolerance(self, tmp_path):
        rec = tmp_path / "Video_2026-07-04_150000.mp4"
        rec.write_bytes(b"fake")
        events = [
            make_event("Standup", datetime(2026, 7, 4, 9, 0), datetime(2026, 7, 4, 9, 15)),
            make_event("Customer Call", datetime(2026, 7, 4, 14, 0), datetime(2026, 7, 4, 15, 5)),
        ]
        rows = ccr.correlate(events, [rec], tolerance_min=45.0)
        assert rows[0]["status"] == "matched"
        assert rows[0]["matched_title"] == "Customer Call"
        assert rows[0]["diff_minutes"] == pytest.approx(5.0, abs=0.1)

    def test_outside_tolerance_is_unmatched(self, tmp_path):
        rec = tmp_path / "Video_2026-07-04_200000.mp4"
        rec.write_bytes(b"fake")
        events = [make_event("Customer Call", datetime(2026, 7, 4, 14, 0), datetime(2026, 7, 4, 15, 0))]
        rows = ccr.correlate(events, [rec], tolerance_min=45.0)
        assert rows[0]["status"] == "no_match_within_tolerance"
        assert rows[0]["matched_title"] is None
        assert rows[0]["diff_minutes"] > 45.0

    def test_no_events_yields_no_calendar_events_status(self, tmp_path):
        rec = tmp_path / "Video_2026-07-04_150000.mp4"
        rec.write_bytes(b"fake")
        rows = ccr.correlate([], [rec], tolerance_min=45.0)
        assert rows[0]["status"] == "no_calendar_events"
        assert rows[0]["diff_minutes"] is None

    def test_events_with_no_end_are_never_matched(self, tmp_path):
        rec = tmp_path / "Video_2026-07-04_150000.mp4"
        rec.write_bytes(b"fake")
        events = [{"title": "Broken", "start": None, "end": None,
                  "all_day": False, "comments": ""}]
        rows = ccr.correlate(events, [rec], tolerance_min=45.0)
        assert rows[0]["status"] == "no_calendar_events"

    def test_multiple_recordings_matched_independently(self, tmp_path):
        rec1 = tmp_path / "Video_2026-07-04_091500.mp4"
        rec1.write_bytes(b"fake")
        rec2 = tmp_path / "Video_2026-07-04_150500.mp4"
        rec2.write_bytes(b"fake")
        events = [
            make_event("Standup", datetime(2026, 7, 4, 9, 0), datetime(2026, 7, 4, 9, 15)),
            make_event("Customer Call", datetime(2026, 7, 4, 14, 0), datetime(2026, 7, 4, 15, 0)),
        ]
        rows = ccr.correlate(events, [rec1, rec2], tolerance_min=45.0)
        by_name = {r["recording"].name: r["matched_title"] for r in rows}
        assert by_name["Video_2026-07-04_091500.mp4"] == "Standup"
        assert by_name["Video_2026-07-04_150500.mp4"] == "Customer Call"

    def test_exact_tolerance_boundary_is_inclusive(self, tmp_path):
        rec = tmp_path / "Video_2026-07-04_154500.mp4"
        rec.write_bytes(b"fake")
        events = [make_event("Call", datetime(2026, 7, 4, 14, 0), datetime(2026, 7, 4, 15, 0))]
        rows = ccr.correlate(events, [rec], tolerance_min=45.0)
        assert rows[0]["status"] == "matched"

    def test_matched_row_carries_attendees_agenda_description(self, tmp_path):
        rec = tmp_path / "Video_2026-07-04_150000.mp4"
        rec.write_bytes(b"fake")
        events = [make_event(
            "Customer Call", datetime(2026, 7, 4, 14, 0), datetime(2026, 7, 4, 15, 0),
            comments="Discuss GC-MS quote.", agenda="1. Intro\n2. Demo",
            attendees=[{"name": "Jane Doe", "email": "jane@example.com"}])]
        rows = ccr.correlate(events, [rec], tolerance_min=45.0)
        assert rows[0]["attendees"] == [{"name": "Jane Doe", "email": "jane@example.com"}]
        assert rows[0]["agenda"] == "1. Intro\n2. Demo"
        assert rows[0]["description"] == "Discuss GC-MS quote."

    def test_unmatched_row_has_blank_attendees_agenda_description(self, tmp_path):
        rec = tmp_path / "Video_2026-07-04_200000.mp4"
        rec.write_bytes(b"fake")
        events = [make_event(
            "Customer Call", datetime(2026, 7, 4, 14, 0), datetime(2026, 7, 4, 15, 0),
            comments="Should not leak.", agenda="Should not leak.",
            attendees=[{"name": "Jane Doe", "email": "jane@example.com"}])]
        rows = ccr.correlate(events, [rec], tolerance_min=45.0)
        assert rows[0]["status"] == "no_match_within_tolerance"
        assert rows[0]["attendees"] == []
        assert rows[0]["agenda"] == ""
        assert rows[0]["description"] == ""


# -----------------------------------------------------------------
# write_report_csv
# -----------------------------------------------------------------

class TestWriteReportCsv:
    def test_writes_expected_columns_and_rows(self, tmp_path):
        rec = tmp_path / "Video_2026-07-04_150000.mp4"
        rec.write_bytes(b"fake")
        rows = [{
            "recording": rec, "recording_end": datetime(2026, 7, 4, 15, 0, 0),
            "time_source": "filename", "matched_title": "Customer Call",
            "event_start": datetime(2026, 7, 4, 14, 0, 0), "event_end": datetime(2026, 7, 4, 15, 0, 0),
            "diff_minutes": 0.0, "status": "matched",
            "attendees": [{"name": "Jane Doe", "email": "jane@example.com"}],
            "agenda": "1. Intro", "description": "Discuss GC-MS quote.",
        }]
        out = tmp_path / "report.csv"
        ccr.write_report_csv(rows, out)

        with open(out, encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            header = next(reader)
            data_row = next(reader)
        assert header == ["Recording", "Recording End Time", "Time Source", "Matched Meeting",
                          "Meeting Start", "Meeting End", "Diff (min)", "Status",
                          "Attendees", "Agenda", "Description"]
        assert data_row[0] == "Video_2026-07-04_150000.mp4"
        assert data_row[3] == "Customer Call"
        assert data_row[7] == "matched"
        assert data_row[8] == "Jane Doe <jane@example.com>"
        assert data_row[9] == "1. Intro"
        assert data_row[10] == "Discuss GC-MS quote."

    def test_unmatched_row_has_blank_meeting_fields(self, tmp_path):
        rec = tmp_path / "orphan.mp4"
        rec.write_bytes(b"fake")
        rows = [{
            "recording": rec, "recording_end": datetime(2026, 7, 4, 15, 0, 0),
            "time_source": "mtime", "matched_title": None, "event_start": None,
            "event_end": None, "diff_minutes": None, "status": "no_calendar_events",
            "attendees": [], "agenda": "", "description": "",
        }]
        out = tmp_path / "report.csv"
        ccr.write_report_csv(rows, out)
        with open(out, encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            next(reader)
            data_row = next(reader)
        assert data_row[3] == ""  # Matched Meeting
        assert data_row[4] == ""  # Meeting Start
        assert data_row[6] == ""  # Diff (min)
        assert data_row[8] == ""  # Attendees
        assert data_row[9] == ""  # Agenda
        assert data_row[10] == ""  # Description

    def test_missing_optional_keys_default_blank(self, tmp_path):
        # A row dict built without attendees/agenda/description keys at
        # all (e.g. hand-built by an older caller) must not KeyError -
        # write_report_csv uses .get() with blank/empty defaults.
        rec = tmp_path / "orphan.mp4"
        rec.write_bytes(b"fake")
        rows = [{
            "recording": rec, "recording_end": datetime(2026, 7, 4, 15, 0, 0),
            "time_source": "mtime", "matched_title": None, "event_start": None,
            "event_end": None, "diff_minutes": None, "status": "no_calendar_events",
        }]
        out = tmp_path / "report.csv"
        ccr.write_report_csv(rows, out)
        with open(out, encoding="utf-8-sig") as f:
            reader = csv.reader(f)
            next(reader)
            data_row = next(reader)
        assert data_row[8] == ""
        assert data_row[9] == ""
        assert data_row[10] == ""

    def test_creates_parent_directories(self, tmp_path):
        out = tmp_path / "nested" / "dir" / "report.csv"
        ccr.write_report_csv([], out)
        assert out.exists()


# -----------------------------------------------------------------
# write_batch_csv (task #96): batch-table-ready CSV the GUI's
# "Load .txt/.csv..." button already recognizes unchanged.
# -----------------------------------------------------------------

class TestWriteBatchCsv:
    def test_writes_expected_header(self, tmp_path):
        out = tmp_path / "batch.csv"
        ccr.write_batch_csv([], out)
        with open(out, encoding="utf-8-sig") as f:
            header = next(csv.reader(f))
        assert header == ["file", "language", "title", "date", "comments"]

    def test_matched_row_has_full_path_title_date(self, tmp_path):
        rec = tmp_path / "Video_2026-07-04_150000.mp4"
        rec.write_bytes(b"fake")
        rows = [{
            "recording": rec, "matched_title": "Customer Call",
            "event_start": datetime(2026, 7, 4, 14, 0, 0),
            "attendees": [], "agenda": "", "description": "",
        }]
        out = tmp_path / "batch.csv"
        ccr.write_batch_csv(rows, out)
        with open(out, encoding="utf-8-sig") as f:
            row = next(csv.DictReader(f))
        assert row["file"] == str(rec)
        assert row["language"] == ""
        assert row["title"] == "Customer Call"
        assert row["date"] == "2026-07-04"

    def test_comments_fold_description_agenda_attendees(self, tmp_path):
        rec = tmp_path / "Video_2026-07-04_150000.mp4"
        rec.write_bytes(b"fake")
        rows = [{
            "recording": rec, "matched_title": "Customer Call",
            "event_start": datetime(2026, 7, 4, 14, 0, 0),
            "attendees": [{"name": "Jane Doe", "email": "jane@example.com"}],
            "agenda": "1. Intro", "description": "Discuss GC-MS quote.",
        }]
        out = tmp_path / "batch.csv"
        ccr.write_batch_csv(rows, out)
        with open(out, encoding="utf-8-sig") as f:
            row = next(csv.DictReader(f))
        assert "Discuss GC-MS quote." in row["comments"]
        assert "Agenda: 1. Intro" in row["comments"]
        assert "Invitees: Jane Doe <jane@example.com>" in row["comments"]

    def test_unmatched_row_has_full_path_but_blank_title_date_comments(self, tmp_path):
        rec = tmp_path / "orphan.mp4"
        rec.write_bytes(b"fake")
        rows = [{
            "recording": rec, "matched_title": None, "event_start": None,
            "attendees": [], "agenda": "", "description": "",
        }]
        out = tmp_path / "batch.csv"
        ccr.write_batch_csv(rows, out)
        with open(out, encoding="utf-8-sig") as f:
            row = next(csv.DictReader(f))
        assert row["file"] == str(rec)
        assert row["title"] == ""
        assert row["date"] == ""
        assert row["comments"] == ""

    def test_header_is_loadable_by_gui_logic_csv_header_index(self, tmp_path):
        # Cross-check against the actual GUI-side matcher (gui_logic.py)
        # rather than just asserting our own header string, so a future
        # rename on either side would break this test, not silently
        # decouple the two.
        import gui_logic
        out = tmp_path / "batch.csv"
        ccr.write_batch_csv([], out)
        with open(out, encoding="utf-8-sig") as f:
            header = next(csv.reader(f))
        known_cols = {"file", "language", "title", "date", "comments", "qa_start", "qa_end"}
        idx = gui_logic.csv_header_index(header, known_cols)
        assert set(idx.keys()) == {"file", "language", "title", "date", "comments"}


# -----------------------------------------------------------------
# main() - end-to-end smoke test with real files on disk
# -----------------------------------------------------------------

class TestMainEndToEnd:
    def test_full_run_produces_report_with_expected_match(self, tmp_path, capsys):
        (tmp_path / "calendar.ics").write_text(
            "BEGIN:VCALENDAR\nBEGIN:VEVENT\nSUMMARY:Customer Call\n"
            "DTSTART:20260704T140000\nDTEND:20260704T150000\n"
            "ATTENDEE;CN=Jane Doe:mailto:jane@example.com\n"
            "DESCRIPTION:Agenda:\\n1. Intro\\n\\nJoin link: https://example.com\n"
            "END:VEVENT\nEND:VCALENDAR\n",
            encoding="utf-8")
        (tmp_path / "Video_2026-07-04_150200.mp4").write_bytes(b"fake")

        # main() reads argv via argparse; call directly with sys.argv patched.
        import sys as _sys
        old_argv = _sys.argv
        try:
            _sys.argv = ["correlate_calendar_recordings.py", str(tmp_path)]
            exit_code = ccr.main()
        finally:
            _sys.argv = old_argv

        assert exit_code == 0
        out_path = tmp_path / "calendar_recording_correlation.csv"
        assert out_path.exists()
        with open(out_path, encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            row = next(reader)
        assert row["Matched Meeting"] == "Customer Call"
        assert row["Status"] == "matched"
        assert row["Attendees"] == "Jane Doe <jane@example.com>"
        assert row["Agenda"] == "1. Intro"
        assert "Join link" in row["Description"]

        # task #96: the batch-ready CSV is written alongside by default.
        batch_path = tmp_path / "calendar_recording_correlation_batch.csv"
        assert batch_path.exists()
        with open(batch_path, encoding="utf-8-sig") as f:
            batch_row = next(csv.DictReader(f))
        assert batch_row["file"] == str(tmp_path / "Video_2026-07-04_150200.mp4")
        assert batch_row["title"] == "Customer Call"
        assert batch_row["date"] == "2026-07-04"
        assert "Invitees: Jane Doe <jane@example.com>" in batch_row["comments"]

    def test_no_batch_csv_flag_skips_it(self, tmp_path):
        (tmp_path / "calendar.ics").write_text(
            "BEGIN:VCALENDAR\nEND:VCALENDAR\n", encoding="utf-8")
        (tmp_path / "Video_2026-07-04_150200.mp4").write_bytes(b"fake")
        import sys as _sys
        old_argv = _sys.argv
        try:
            _sys.argv = ["correlate_calendar_recordings.py", str(tmp_path), "--no-batch-csv"]
            exit_code = ccr.main()
        finally:
            _sys.argv = old_argv
        assert exit_code == 0
        assert not (tmp_path / "calendar_recording_correlation_batch.csv").exists()

    def test_missing_folder_returns_nonzero(self, tmp_path):
        import sys as _sys
        old_argv = _sys.argv
        try:
            _sys.argv = ["correlate_calendar_recordings.py", str(tmp_path / "does_not_exist")]
            exit_code = ccr.main()
        finally:
            _sys.argv = old_argv
        assert exit_code == 1

    def test_no_ics_found_returns_nonzero(self, tmp_path):
        (tmp_path / "Video_2026-07-04_150000.mp4").write_bytes(b"fake")
        import sys as _sys
        old_argv = _sys.argv
        try:
            _sys.argv = ["correlate_calendar_recordings.py", str(tmp_path)]
            exit_code = ccr.main()
        finally:
            _sys.argv = old_argv
        assert exit_code == 1

    def test_no_recordings_found_returns_nonzero(self, tmp_path):
        (tmp_path / "calendar.ics").write_text(
            "BEGIN:VCALENDAR\nEND:VCALENDAR\n", encoding="utf-8")
        import sys as _sys
        old_argv = _sys.argv
        try:
            _sys.argv = ["correlate_calendar_recordings.py", str(tmp_path)]
            exit_code = ccr.main()
        finally:
            _sys.argv = old_argv
        assert exit_code == 1
