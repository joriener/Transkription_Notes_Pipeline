# =============================================================
#  Transkription_Notes_Pipeline - db.py
#  Unified SQLite schema merging the transcript log from
#  Audio_Transkription_Notes_Pipeline and the slide/run_log
#  tables from Video_Transkription_Notes_Pipeline.
#
#  V1.2: notes_text column + FTS5 full-text search over notes and
#  slides, database management helpers (create/rename/stats), and
#  a resume-lookup helper for --file-list batch mode.
#
#  Usage:
#    python db.py                 (print summary of all processed files)
#    python db.py --search "retention index"   (full-text search)
#    from db import get_connection, validate_schema, upsert_transcript, ...
# =============================================================

import sqlite3
import logging
from pathlib import Path

log = logging.getLogger(__name__)

SCHEMA_VERSION = 3

CREATE_TRANSCRIPTS = """
CREATE TABLE IF NOT EXISTS transcripts (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    file_path        TEXT    NOT NULL UNIQUE,
    file_name        TEXT    NOT NULL,
    whisper_model    TEXT,
    language         TEXT,
    segment_count    INTEGER,
    duration_sec     REAL,
    notes_generated  INTEGER DEFAULT 0,
    notes_backend    TEXT,
    prompt_template  TEXT,
    notes_text       TEXT    DEFAULT '',
    processed_at     TEXT    DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_transcripts_file ON transcripts(file_path);
"""

CREATE_SLIDES = """
CREATE TABLE IF NOT EXISTS slides (
    slide_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    video_path      TEXT    NOT NULL,
    timestamp_sec   REAL    NOT NULL,
    snapshot_path   TEXT,
    hash_value      TEXT,
    title           TEXT,
    bullets         TEXT,    -- JSON array string
    slide_type      TEXT,
    transcript_seg  TEXT,
    speaker         TEXT,
    created_at      TEXT    DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_slides_video ON slides(video_path);
"""

CREATE_RUN_LOG = """
CREATE TABLE IF NOT EXISTS run_log (
    run_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at  TEXT    DEFAULT (datetime('now')),
    source_path TEXT,
    mode        TEXT,     -- "audio" or "video"
    slides_found INTEGER,
    status      TEXT,
    notes       TEXT
);
"""

CREATE_META = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""

# ---------------------------------------------------------------------------
# Known speakers (task #79): a global voiceprint roster, shared across every
# recording the pipeline processes. embedding is a float32 vector (see
# speaker_id.serialize_embedding/deserialize_embedding) stored as raw bytes;
# embedding_dim is kept alongside it so a future switch to a different
# embedding model with a different vector size fails loudly instead of
# silently comparing incompatible vectors. sample_count tracks how many
# confirmed renames have contributed to the stored embedding via the
# running-average update in speaker_id.update_running_average.
# ---------------------------------------------------------------------------

CREATE_KNOWN_SPEAKERS = """
CREATE TABLE IF NOT EXISTS known_speakers (
    speaker_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT    NOT NULL UNIQUE,
    embedding     BLOB    NOT NULL,
    embedding_dim INTEGER NOT NULL,
    sample_count  INTEGER DEFAULT 1,
    created_at    TEXT    DEFAULT (datetime('now')),
    updated_at    TEXT    DEFAULT (datetime('now'))
);
"""

# ---------------------------------------------------------------------------
# FTS5 full-text search over notes (transcripts.notes_text) and slides
# (title/bullets/transcript_seg). External-content tables keyed on the
# rowid of the source table, kept in sync via triggers. If the sqlite3
# build in use lacks FTS5 (rare), creation fails gracefully and search
# falls back to a plain LIKE query (see FTS5_AVAILABLE below).
# ---------------------------------------------------------------------------

CREATE_NOTES_FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(
    file_path UNINDEXED,
    file_name,
    notes_text,
    content='transcripts',
    content_rowid='id'
);
CREATE TRIGGER IF NOT EXISTS transcripts_ai AFTER INSERT ON transcripts BEGIN
    INSERT INTO notes_fts(rowid, file_path, file_name, notes_text)
    VALUES (new.id, new.file_path, new.file_name, new.notes_text);
END;
CREATE TRIGGER IF NOT EXISTS transcripts_ad AFTER DELETE ON transcripts BEGIN
    INSERT INTO notes_fts(notes_fts, rowid, file_path, file_name, notes_text)
    VALUES('delete', old.id, old.file_path, old.file_name, old.notes_text);
END;
CREATE TRIGGER IF NOT EXISTS transcripts_au AFTER UPDATE ON transcripts BEGIN
    INSERT INTO notes_fts(notes_fts, rowid, file_path, file_name, notes_text)
    VALUES('delete', old.id, old.file_path, old.file_name, old.notes_text);
    INSERT INTO notes_fts(rowid, file_path, file_name, notes_text)
    VALUES (new.id, new.file_path, new.file_name, new.notes_text);
END;
"""

CREATE_SLIDES_FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS slides_fts USING fts5(
    video_path UNINDEXED,
    title,
    bullets,
    transcript_seg,
    content='slides',
    content_rowid='slide_id'
);
CREATE TRIGGER IF NOT EXISTS slides_ai AFTER INSERT ON slides BEGIN
    INSERT INTO slides_fts(rowid, video_path, title, bullets, transcript_seg)
    VALUES (new.slide_id, new.video_path, new.title, new.bullets, new.transcript_seg);
END;
CREATE TRIGGER IF NOT EXISTS slides_ad AFTER DELETE ON slides BEGIN
    INSERT INTO slides_fts(slides_fts, rowid, video_path, title, bullets, transcript_seg)
    VALUES('delete', old.slide_id, old.video_path, old.title, old.bullets, old.transcript_seg);
END;
CREATE TRIGGER IF NOT EXISTS slides_au AFTER UPDATE ON slides BEGIN
    INSERT INTO slides_fts(slides_fts, rowid, video_path, title, bullets, transcript_seg)
    VALUES('delete', old.slide_id, old.video_path, old.title, old.bullets, old.transcript_seg);
    INSERT INTO slides_fts(rowid, video_path, title, bullets, transcript_seg)
    VALUES (new.slide_id, new.video_path, new.title, new.bullets, new.transcript_seg);
END;
"""

# Set at runtime by validate_schema(); read by search_notes()/search_slides()
# to decide between FTS5 MATCH and a plain LIKE fallback.
FTS5_AVAILABLE = True


def get_connection(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    # DELETE journal mode instead of WAL (2026-07-02): WAL keeps a separate
    # -wal/-shm file that must stay byte-for-byte in sync with the main .db
    # between writes. This project's db lives in a folder that is mirrored
    # by an external sync process (Cowork mount); if that process reads or
    # copies the .db file mid-checkpoint, the -wal file goes out of sync
    # with the main file and sqlite reports "database disk image is
    # malformed" on the next open, exactly the corruption seen twice in
    # this project already, including immediately after a fresh rebuild.
    # DELETE mode removes the rollback journal after every commit instead
    # of accumulating a separate WAL file, so there is nothing that can
    # desync between writes. Slightly slower under heavy concurrent access,
    # irrelevant for a single-user desktop GUI.
    conn.execute("PRAGMA journal_mode=DELETE;")
    return conn


def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    cur = conn.execute(f"PRAGMA table_info({table})")
    return any(row["name"] == column for row in cur.fetchall())


def validate_schema(db_path: str) -> bool:
    """Run before every operation. Creates tables if missing, migrates
    older databases (adds notes_text column, builds FTS5 indexes)."""
    global FTS5_AVAILABLE
    try:
        conn = get_connection(db_path)
        conn.executescript(CREATE_TRANSCRIPTS + CREATE_SLIDES + CREATE_RUN_LOG + CREATE_META
                            + CREATE_KNOWN_SPEAKERS)

        # Migration: older databases (schema v1) lack notes_text.
        if not _column_exists(conn, "transcripts", "notes_text"):
            conn.execute("ALTER TABLE transcripts ADD COLUMN notes_text TEXT DEFAULT ''")
            log.info("DB migration: added transcripts.notes_text column.")

        try:
            conn.executescript(CREATE_NOTES_FTS + CREATE_SLIDES_FTS)
            FTS5_AVAILABLE = True
        except sqlite3.OperationalError as exc:
            FTS5_AVAILABLE = False
            log.warning("FTS5 not available in this sqlite3 build (%s). "
                        "Search will fall back to a plain LIKE query.", exc)

        conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            ("schema_version", str(SCHEMA_VERSION)),
        )
        conn.commit()
        conn.close()
        log.info("Schema validated: %s", db_path)
        return True
    except Exception as exc:
        log.error("Schema validation failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Transcripts
# ---------------------------------------------------------------------------

def upsert_transcript(conn: sqlite3.Connection, file_path: str, whisper_model: str,
                      language: str, segment_count: int, duration_sec: float,
                      notes_generated: bool = False, notes_backend: str = "",
                      prompt_template: str = "", notes_text: str = "") -> None:
    """Insert or update transcript metadata, keyed on file_path.
    notes_text (full generated notes markdown) feeds the FTS5 search index;
    pass "" to leave it out of a whisper-only update."""
    conn.execute(
        """INSERT INTO transcripts
           (file_path, file_name, whisper_model, language, segment_count,
            duration_sec, notes_generated, notes_backend, prompt_template, notes_text)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(file_path) DO UPDATE SET
               whisper_model=excluded.whisper_model,
               language=excluded.language,
               segment_count=excluded.segment_count,
               duration_sec=excluded.duration_sec,
               notes_generated=excluded.notes_generated,
               notes_backend=excluded.notes_backend,
               prompt_template=excluded.prompt_template,
               notes_text=CASE WHEN excluded.notes_text != '' THEN excluded.notes_text
                               ELSE transcripts.notes_text END,
               processed_at=datetime('now')""",
        (file_path, Path(file_path).name, whisper_model, language,
         segment_count, round(duration_sec, 1) if duration_sec else 0,
         int(notes_generated), notes_backend, prompt_template, notes_text),
    )
    conn.commit()
    log.info("DB: logged %s (%d segments, %.0fs).",
             Path(file_path).name, segment_count, duration_sec or 0)


def list_transcripts(conn: sqlite3.Connection) -> list[dict]:
    cur = conn.execute("SELECT * FROM transcripts ORDER BY processed_at DESC")
    return [dict(row) for row in cur.fetchall()]


def get_completed_transcript(conn: sqlite3.Connection, file_path: str) -> dict | None:
    """Return the transcripts row for file_path if notes were already
    generated for it (used by --file-list resume/retry), else None."""
    cur = conn.execute(
        "SELECT * FROM transcripts WHERE file_path = ? AND notes_generated = 1",
        (file_path,),
    )
    row = cur.fetchone()
    return dict(row) if row else None


def update_source_path(conn: sqlite3.Connection, old_path: str, new_path: str) -> None:
    """
    Repoint every DB row that references old_path to new_path - used when
    move_processed_files (see config.py) relocates a source file to its
    processed_subfolder_name after a successful run. Updates both
    transcripts.file_path/file_name (keyed UNIQUE on file_path, so this is
    an UPDATE not another upsert) and slides.video_path (video mode only,
    no UNIQUE constraint there so a plain UPDATE across all matching rows
    is safe). A no-op (0 rows affected) is not an error: audio-only runs
    have no slides rows, and a --no-whisper run has no transcripts row
    either.
    """
    conn.execute(
        "UPDATE transcripts SET file_path = ?, file_name = ? WHERE file_path = ?",
        (new_path, Path(new_path).name, old_path),
    )
    conn.execute(
        "UPDATE slides SET video_path = ? WHERE video_path = ?",
        (new_path, old_path),
    )
    conn.commit()
    log.info("DB: source path updated after move: %s -> %s", old_path, new_path)


# ---------------------------------------------------------------------------
# Known speakers (task #79)
# ---------------------------------------------------------------------------

def insert_known_speaker(conn: sqlite3.Connection, name: str, embedding: bytes,
                          embedding_dim: int) -> int:
    """Enroll a brand-new speaker voiceprint. Raises sqlite3.IntegrityError
    if name already exists (callers should check get_known_speaker_by_name
    first, or catch this, since the UNIQUE constraint is the source of
    truth for name collisions)."""
    cur = conn.execute(
        """INSERT INTO known_speakers (name, embedding, embedding_dim, sample_count)
           VALUES (?, ?, ?, 1)""",
        (name, embedding, embedding_dim),
    )
    conn.commit()
    log.info("DB: enrolled new known speaker '%s' (dim=%d).", name, embedding_dim)
    return cur.lastrowid


def list_known_speakers(conn: sqlite3.Connection) -> list[dict]:
    cur = conn.execute("SELECT * FROM known_speakers ORDER BY name COLLATE NOCASE")
    return [dict(row) for row in cur.fetchall()]


def get_known_speaker_by_name(conn: sqlite3.Connection, name: str) -> dict | None:
    cur = conn.execute("SELECT * FROM known_speakers WHERE name = ?", (name,))
    row = cur.fetchone()
    return dict(row) if row else None


def get_known_speaker(conn: sqlite3.Connection, speaker_id: int) -> dict | None:
    cur = conn.execute("SELECT * FROM known_speakers WHERE speaker_id = ?", (speaker_id,))
    row = cur.fetchone()
    return dict(row) if row else None


def update_known_speaker_embedding(conn: sqlite3.Connection, speaker_id: int,
                                    embedding: bytes, sample_count: int) -> None:
    """Overwrite the stored voiceprint after a confirmed match (caller has
    already computed the new running-average vector and sample_count via
    speaker_id.update_running_average)."""
    conn.execute(
        """UPDATE known_speakers
           SET embedding = ?, sample_count = ?, updated_at = datetime('now')
           WHERE speaker_id = ?""",
        (embedding, sample_count, speaker_id),
    )
    conn.commit()


def rename_known_speaker(conn: sqlite3.Connection, speaker_id: int, new_name: str) -> None:
    """Raises sqlite3.IntegrityError if new_name collides with a different
    existing speaker (UNIQUE constraint)."""
    conn.execute(
        "UPDATE known_speakers SET name = ?, updated_at = datetime('now') WHERE speaker_id = ?",
        (new_name, speaker_id),
    )
    conn.commit()
    log.info("DB: renamed known speaker #%d to '%s'.", speaker_id, new_name)


def delete_known_speaker(conn: sqlite3.Connection, speaker_id: int) -> None:
    """Removes a voiceprint from the roster. Does not touch any already-
    written transcript/segment files; future runs simply stop suggesting
    this name."""
    conn.execute("DELETE FROM known_speakers WHERE speaker_id = ?", (speaker_id,))
    conn.commit()
    log.info("DB: deleted known speaker #%d.", speaker_id)


# ---------------------------------------------------------------------------
# Slides (video mode only)
# ---------------------------------------------------------------------------

def insert_slide(conn: sqlite3.Connection, record: dict) -> int:
    cur = conn.execute(
        """INSERT INTO slides
           (video_path, timestamp_sec, snapshot_path, hash_value,
            title, bullets, slide_type, transcript_seg, speaker)
           VALUES (:video_path, :timestamp_sec, :snapshot_path, :hash_value,
                   :title, :bullets, :slide_type, :transcript_seg, :speaker)""",
        record,
    )
    conn.commit()
    return cur.lastrowid


def update_slide_annotation(conn: sqlite3.Connection, video_path: str, timestamp_sec: float,
                            title: str, bullets_json: str, slide_type: str) -> bool:
    """
    Update title/bullets/slide_type for an existing slide row, matched on
    (video_path, timestamp_sec) rather than inserting a new row. Used by
    the "re-annotate failed slides" flow so a retry never creates a
    duplicate slide entry. Returns True if a row was actually updated.
    """
    cur = conn.execute(
        """UPDATE slides SET title = ?, bullets = ?, slide_type = ?
           WHERE video_path = ? AND timestamp_sec = ?""",
        (title, bullets_json, slide_type, video_path, timestamp_sec),
    )
    conn.commit()
    return cur.rowcount > 0


def get_slides_for_video(conn: sqlite3.Connection, video_path: str) -> list:
    cur = conn.execute(
        "SELECT * FROM slides WHERE video_path = ? ORDER BY timestamp_sec",
        (video_path,),
    )
    return [dict(row) for row in cur.fetchall()]


# ---------------------------------------------------------------------------
# Run log
# ---------------------------------------------------------------------------

def insert_run_log(conn: sqlite3.Connection, source_path: str, mode: str,
                   slides_found: int, status: str, notes: str = "") -> int:
    cur = conn.execute(
        """INSERT INTO run_log (source_path, mode, slides_found, status, notes)
           VALUES (?, ?, ?, ?, ?)""",
        (source_path, mode, slides_found, status, notes),
    )
    conn.commit()
    return cur.lastrowid


# ---------------------------------------------------------------------------
# Full-text search (FTS5, with LIKE fallback if FTS5 is unavailable)
# ---------------------------------------------------------------------------

def _fts_query(raw: str) -> str:
    """FTS5 MATCH syntax errors on many special characters. Treat the
    whole query as a literal phrase so 'retention index' searches for
    that exact phrase rather than being parsed as FTS5 query syntax."""
    escaped = raw.replace('"', '""')
    return f'"{escaped}"'


def search_notes(conn: sqlite3.Connection, query: str, limit: int = 50) -> list[dict]:
    """Search generated notes text. Returns file_path, file_name,
    processed_at, and a highlighted snippet for each match."""
    query = (query or "").strip()
    if not query:
        return []

    if FTS5_AVAILABLE:
        try:
            cur = conn.execute(
                """SELECT t.file_path, t.file_name, t.processed_at, t.prompt_template,
                          snippet(notes_fts, 2, '[', ']', ' ... ', 12) AS snippet
                   FROM notes_fts
                   JOIN transcripts t ON t.id = notes_fts.rowid
                   WHERE notes_fts MATCH ?
                   ORDER BY rank
                   LIMIT ?""",
                (_fts_query(query), limit),
            )
            return [dict(row) for row in cur.fetchall()]
        except sqlite3.OperationalError as exc:
            log.warning("FTS5 search failed (%s), falling back to LIKE.", exc)

    like = f"%{query}%"
    cur = conn.execute(
        """SELECT file_path, file_name, processed_at, prompt_template,
                  substr(notes_text, 1, 200) AS snippet
           FROM transcripts
           WHERE notes_text LIKE ? OR file_name LIKE ?
           ORDER BY processed_at DESC
           LIMIT ?""",
        (like, like, limit),
    )
    return [dict(row) for row in cur.fetchall()]


def search_slides(conn: sqlite3.Connection, query: str, limit: int = 50) -> list[dict]:
    """Search slide titles/bullets/aligned transcript segments."""
    query = (query or "").strip()
    if not query:
        return []

    if FTS5_AVAILABLE:
        try:
            cur = conn.execute(
                """SELECT s.video_path, s.timestamp_sec, s.title, s.slide_type,
                          snippet(slides_fts, 2, '[', ']', ' ... ', 12) AS snippet
                   FROM slides_fts
                   JOIN slides s ON s.slide_id = slides_fts.rowid
                   WHERE slides_fts MATCH ?
                   ORDER BY rank
                   LIMIT ?""",
                (_fts_query(query), limit),
            )
            return [dict(row) for row in cur.fetchall()]
        except sqlite3.OperationalError as exc:
            log.warning("FTS5 search failed (%s), falling back to LIKE.", exc)

    like = f"%{query}%"
    cur = conn.execute(
        """SELECT video_path, timestamp_sec, title, slide_type,
                  substr(bullets, 1, 200) AS snippet
           FROM slides
           WHERE title LIKE ? OR bullets LIKE ? OR transcript_seg LIKE ?
           ORDER BY video_path, timestamp_sec
           LIMIT ?""",
        (like, like, like, limit),
    )
    return [dict(row) for row in cur.fetchall()]


def search_all(db_path: str, query: str, limit: int = 50) -> dict:
    """Convenience entry point for CLI/GUI: opens the connection, runs
    both searches, closes the connection, returns {"notes": [...], "slides": [...]}."""
    validate_schema(db_path)
    conn = get_connection(db_path)
    try:
        return {
            "notes": search_notes(conn, query, limit),
            "slides": search_slides(conn, query, limit),
        }
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Database file management (open/new/rename) - backs the GUI Settings tab
# ---------------------------------------------------------------------------

def create_new_db(path: str) -> str:
    """Create a fresh, empty database with the current schema at path.
    Raises FileExistsError if a file is already there."""
    p = Path(path)
    if p.exists():
        raise FileExistsError(f"{path} already exists. Choose a new filename or use 'Open existing'.")
    if not validate_schema(str(p)):
        raise RuntimeError(f"Could not initialise schema at {path}.")
    log.info("New database created: %s", path)
    return str(p)


def rename_db(old_path: str, new_path: str) -> str:
    """Rename/move a database file, including its WAL/SHM/journal
    sidecar files if present. Only safe to call between runs, while no
    connection to old_path is open."""
    old, new = Path(old_path), Path(new_path)
    if not old.exists():
        raise FileNotFoundError(f"{old_path} does not exist.")
    if new.exists():
        raise FileExistsError(f"{new_path} already exists.")
    new.parent.mkdir(parents=True, exist_ok=True)
    old.rename(new)
    for suffix in ("-wal", "-shm", "-journal"):
        side = Path(str(old) + suffix)
        if side.exists():
            side.rename(str(new) + suffix)
    log.info("Database renamed: %s -> %s", old_path, new_path)
    return str(new)


def get_db_stats(db_path: str) -> dict:
    """Summary used by the GUI Settings tab to show what's in the
    currently selected database without opening a full connection UI."""
    p = Path(db_path)
    if not p.exists():
        return {"exists": False, "path": db_path}
    conn = get_connection(db_path)
    try:
        n_transcripts = conn.execute("SELECT COUNT(*) FROM transcripts").fetchone()[0]
        n_notes = conn.execute("SELECT COUNT(*) FROM transcripts WHERE notes_generated=1").fetchone()[0]
        n_slides = conn.execute("SELECT COUNT(*) FROM slides").fetchone()[0]
        n_runs = conn.execute("SELECT COUNT(*) FROM run_log").fetchone()[0]
    finally:
        conn.close()
    return {
        "exists": True, "path": str(p.resolve()),
        "size_bytes": p.stat().st_size,
        "transcripts": n_transcripts, "notes_generated": n_notes,
        "slides": n_slides, "runs": n_runs,
        "fts5_available": FTS5_AVAILABLE,
    }


# ---------------------------------------------------------------------------
# CLI: python db.py -> print summary of processed files
#      python db.py --search "query" -> full-text search notes + slides
# ---------------------------------------------------------------------------

def print_summary(db_path: str) -> None:
    if not Path(db_path).exists():
        print(f"  No database found yet: {db_path}")
        return
    validate_schema(db_path)
    conn = get_connection(db_path)
    rows = list_transcripts(conn)
    conn.close()
    if not rows:
        print("  No files logged yet.")
        return
    print(f"\n  {'File':<40} {'Model':<12} {'Lang':<6} {'Segs':>5} {'Min':>6} {'Notes'}")
    print(f"  {'-'*80}")
    for r in rows:
        mins = f"{r['duration_sec']/60:.1f}" if r['duration_sec'] else "-"
        notes = r['notes_backend'] if r['notes_generated'] else "-"
        print(f"  {r['file_name'][:39]:<40} {(r['whisper_model'] or '-')[:11]:<12} "
              f"{(r['language'] or '-')[:5]:<6} {r['segment_count'] or 0:>5} "
              f"{mins:>6} {notes}")


def print_search_results(db_path: str, query: str) -> None:
    results = search_all(db_path, query)
    print(f'\n  Search: "{query}"')
    print(f"  {'-'*80}")
    if not results["notes"] and not results["slides"]:
        print("  No matches.")
        return
    if results["notes"]:
        print(f"\n  Notes ({len(results['notes'])} match(es)):")
        for r in results["notes"]:
            print(f"    {r['file_name']}  ({r['processed_at']})")
            print(f"      {r['snippet']}")
    if results["slides"]:
        print(f"\n  Slides ({len(results['slides'])} match(es)):")
        for r in results["slides"]:
            print(f"    {Path(r['video_path']).name}  t={r['timestamp_sec']:.1f}s  \"{r['title']}\"")
            print(f"      {r['snippet']}")


if __name__ == "__main__":
    import sys
    from config import CONFIG
    if len(sys.argv) > 2 and sys.argv[1] == "--search":
        print_search_results(CONFIG["db_path"], " ".join(sys.argv[2:]))
    else:
        print_summary(CONFIG["db_path"])
