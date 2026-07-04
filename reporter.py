# =============================================================
#  Transkription_Notes_Pipeline - reporter.py
#  All output writers: notes txt/html/docx, slide CSV/JSON/HTML/PDF
#  report, and HTML-to-PDF conversion. Transcript files
#  (_transcript_speakers.txt, _text.txt, _transcript.srt) are
#  written by transcriber.write_transcript_files() instead, so
#  the same three files are produced consistently in both
#  audio-only and video-with-slides mode.
# =============================================================

import csv
import json
import logging
import re
from datetime import datetime, timedelta
from pathlib import Path

log = logging.getLogger(__name__)


def format_ts(seconds: float) -> str:
    """Convert seconds to HH:MM:SS string."""
    return str(timedelta(seconds=int(seconds)))


_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


def _first_sentence(text: str) -> str:
    """Return just the first sentence of text (split on . ! ?), or the
    whole string if no sentence boundary is found. Used by the slide
    report's "first sentence only" transcript mode so a slide with a
    long spoken segment still fits on one page."""
    text = (text or "").strip()
    if not text:
        return ""
    parts = _SENTENCE_END.split(text, maxsplit=1)
    return parts[0].strip()


# ---------------------------------------------------------------------------
# Notes / summary output (meeting or webinar mode, any prompt template)
# ---------------------------------------------------------------------------

def save_notes_txt(notes_text: str, output_path: Path, filename: str,
                   llm_backend: str, model_name: str = "", event_date: str = "",
                   comments: str = "") -> None:
    """Write the raw LLM notes output as plain text with a header.
    event_date (optional): meeting/event date, shown alongside the
    generation timestamp when set via the GUI's Meeting info fields.
    comments (optional): free-text from the GUI's Meeting info
    "Comments" field, shown right below the date."""
    output_path = Path(output_path)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"Notes: {filename}\n")
        if event_date:
            f.write(f"Meeting date: {event_date}\n")
        if comments:
            f.write(f"Comments: {comments}\n")
        f.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
        f.write(f"LLM backend: {llm_backend}")
        if model_name:
            f.write(f" ({model_name})")
        f.write("\n" + "=" * 60 + "\n\n")
        f.write(notes_text)
    log.info("Notes text saved: %s", output_path)


def _parse_notes_sections(md: str) -> list[dict]:
    """
    Shared parser for notes markdown: splits into sections keyed by
    '## HEADING' lines, each holding a list of {"type", "text"} items
    ("bullet" | "quote" | "text"). Used by both the HTML and docx writers
    so their rendering stays in sync.
    """
    sections: list[dict] = []
    current = None

    for line in md.splitlines():
        stripped = line.strip()

        if stripped.startswith("## "):
            if current:
                sections.append(current)
            current = {"title": stripped[3:].strip(), "items": []}
        elif not current:
            continue
        elif stripped.startswith("- "):
            item = stripped[2:].strip()
            item = re.sub(r"^(Q:|A:)\s*", r"\1 ", item)
            current["items"].append({"type": "bullet", "text": item})
        elif stripped.startswith('"') and stripped.endswith('"') and len(stripped) > 1:
            current["items"].append({"type": "quote", "text": stripped})
        elif stripped:
            current["items"].append({"type": "text", "text": stripped})

    if current:
        sections.append(current)
    return sections


def _markdown_to_html(md: str) -> str:
    """
    Minimal Markdown-to-HTML converter for notes/summary output.
    Handles: ## headings, - bullet lists, "quoted" lines, Q:/A: bolding,
    blank lines. No external dependency.
    """
    html_lines = []
    in_list = False

    for line in md.splitlines():
        stripped = line.strip()

        if stripped.startswith("## "):
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            heading = stripped[3:].strip()
            html_lines.append(f"<h2>{heading}</h2>")

        elif stripped.startswith("- "):
            if not in_list:
                html_lines.append("<ul>")
                in_list = True
            item = stripped[2:].strip()
            item = re.sub(r"^(Q:|A:)", r"<strong>\1</strong>", item)
            html_lines.append(f"  <li>{item}</li>")

        elif stripped.startswith('"') and stripped.endswith('"') and len(stripped) > 1:
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            html_lines.append(f"<blockquote>{stripped}</blockquote>")

        elif stripped == "":
            if in_list:
                html_lines.append("</ul>")
                in_list = False

        else:
            if in_list:
                html_lines.append("</ul>")
                in_list = False
            html_lines.append(f"<p>{stripped}</p>")

    if in_list:
        html_lines.append("</ul>")

    return "\n".join(html_lines)


def save_notes_html(notes_text: str, output_path: Path, filename: str,
                    title: str = "MEETING NOTES", generated_by: str = "",
                    event_date: str = "", comments: str = "") -> Path:
    """Write LLM notes/summary as a styled, print-friendly HTML document.
    event_date (optional): meeting/event date, shown in the meta bar
    alongside the filename when set via the GUI's Meeting info fields.
    comments (optional): free-text from the GUI's Meeting info
    "Comments" field, shown below the date."""
    output_path = Path(output_path)
    body = _markdown_to_html(notes_text)
    event_date_html = f'<div class="event-date">Meeting date: {event_date}</div>' if event_date else ""
    comments_html = f'<div class="comments">Comments: {comments}</div>' if comments else ""

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}: {filename}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: Arial, Helvetica, sans-serif;
    line-height: 1.6;
    color: #333;
    max-width: 210mm;
    margin: 0 auto;
    padding: 20mm;
    background: #fff;
  }}
  .header {{ border-bottom: 3px solid #1F5C99; padding-bottom: 12px; margin-bottom: 24px; }}
  .header h1 {{ color: #1F5C99; font-size: 28px; font-weight: bold; margin-bottom: 8px; }}
  .meta {{ display: flex; justify-content: space-between; align-items: center; margin-top: 12px; }}
  .meta .filename {{ font-size: 18px; font-weight: bold; color: #333; }}
  .meta .date {{ font-size: 14px; color: #666; }}
  .event-date {{ font-size: 14px; color: #1F5C99; font-weight: bold; margin-top: 4px; }}
  .comments {{ font-size: 13px; color: #555; margin-top: 4px; font-style: italic; }}
  h2 {{
    background-color: #D6E4F0; color: #1F5C99; font-size: 18px; font-weight: bold;
    padding: 8px 12px; margin: 24px 0 12px; border-left: 4px solid #1F5C99;
  }}
  ul {{ list-style-type: disc; margin-left: 36px; margin-bottom: 12px; }}
  li {{ margin-bottom: 8px; line-height: 1.5; }}
  p {{ margin-bottom: 12px; line-height: 1.5; }}
  blockquote {{ border-left: 3px solid #aaa; margin: .6rem 0 .6rem 1rem; padding: .2rem .8rem; color: #444; font-style: italic; }}
  .footer {{ font-size: .75rem; color: #999; margin-top: 2rem; border-top: 1px solid #eee; padding-top: .5rem; }}
  @media print {{
    body {{ max-width: 100%; padding: 15mm; }}
    h2 {{ page-break-inside: avoid; }}
  }}
</style>
</head>
<body>
  <div class="header">
    <h1>{title}</h1>
    <div class="meta">
      <span class="filename">{filename}</span>
      <span class="date">{datetime.now().strftime("%Y-%m-%d %H:%M")}</span>
    </div>
    {event_date_html}
    {comments_html}
  </div>
  <div class="content">
{body}
  </div>
  <div class="footer">Generated by Transkription_Notes_Pipeline{(" via " + generated_by) if generated_by else ""}</div>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    log.info("Notes HTML saved: %s", output_path)
    return output_path


def save_notes_docx(notes_text: str, output_path: Path, filename: str,
                    title: str = "MEETING NOTES", generated_by: str = "",
                    event_date: str = "", comments: str = "",
                    transcript_text: str = "") -> Path | None:
    """
    Write LLM notes/summary as a real Word document via python-docx.
    Mirrors save_notes_html's section/bullet parsing (_parse_notes_sections)
    so both formats always show the same structure.
    event_date (optional): meeting/event date, shown in the meta line
    when set via the GUI's Meeting info fields.
    comments (optional): free-text from the GUI's Meeting info
    "Comments" field, shown below the date.
    transcript_text (optional): when non-empty (docx_include_transcript
    config flag), the full raw transcript is appended as a final section
    on a fresh page, after the LLM notes.
    Returns the output path, or None if python-docx is not installed.
    """
    try:
        from docx import Document
        from docx.shared import Pt, RGBColor, Cm
        from docx.enum.text import WD_ALIGN_PARAGRAPH
    except ImportError:
        log.error("python-docx not installed. Install with: pip install python-docx")
        return None

    output_path = Path(output_path)
    doc = Document()

    # Title
    h = doc.add_heading(title, level=0)
    h.runs[0].font.color.rgb = RGBColor(0x1F, 0x5C, 0x99)

    # Meta line: source filename (explicit label) + generated-at timestamp.
    meta = doc.add_paragraph()
    source_run = meta.add_run(f"Source file: {filename}    ")
    source_run.bold = True
    date_run = meta.add_run(datetime.now().strftime("%Y-%m-%d %H:%M"))
    date_run.font.color.rgb = RGBColor(0x66, 0x66, 0x66)
    date_run.font.size = Pt(10)

    if event_date:
        event_p = doc.add_paragraph()
        event_run = event_p.add_run(f"Meeting date: {event_date}")
        event_run.bold = True
        event_run.font.color.rgb = RGBColor(0x1F, 0x5C, 0x99)

    if comments:
        comments_p = doc.add_paragraph()
        comments_run = comments_p.add_run(f"Comments: {comments}")
        comments_run.italic = True
        comments_run.font.color.rgb = RGBColor(0x55, 0x55, 0x55)

    for section in _parse_notes_sections(notes_text):
        sh = doc.add_heading(section["title"], level=1)
        sh.runs[0].font.color.rgb = RGBColor(0x1F, 0x5C, 0x99)
        for item in section["items"]:
            if item["type"] == "bullet":
                doc.add_paragraph(item["text"], style="List Bullet")
            elif item["type"] == "quote":
                p = doc.add_paragraph()
                run = p.add_run(item["text"])
                run.italic = True
            else:
                doc.add_paragraph(item["text"])

    if transcript_text.strip():
        doc.add_page_break()
        th = doc.add_heading("Full Transcript", level=1)
        th.runs[0].font.color.rgb = RGBColor(0x1F, 0x5C, 0x99)
        for line in transcript_text.splitlines():
            if line.strip():
                doc.add_paragraph(line)
            else:
                doc.add_paragraph()

    footer_p = doc.add_paragraph()
    footer_run = footer_p.add_run(
        f"Generated by Transkription_Notes_Pipeline{(' via ' + generated_by) if generated_by else ''}"
    )
    footer_run.font.size = Pt(8)
    footer_run.font.color.rgb = RGBColor(0x99, 0x99, 0x99)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_path))
    log.info("Notes DOCX saved: %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Slide report output (video mode only)
# ---------------------------------------------------------------------------

def save_csv(slides: list[dict], output_path: Path) -> None:
    """Write slide index as CSV."""
    fields = [
        "slide_id", "timestamp", "timestamp_sec", "title",
        "slide_type", "bullets", "transcript_seg", "snapshot_path",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for i, slide in enumerate(slides, start=1):
            row = dict(slide)
            row["slide_id"] = i
            row["timestamp"] = format_ts(slide.get("timestamp_sec", 0))
            bullets = slide.get("bullets", [])
            # Defensive: coerce every item to str regardless of source, so a
            # malformed upstream value (e.g. a weaker VLM nesting bullets as
            # objects/sub-lists) can never crash the whole run at export time.
            row["bullets"] = " | ".join(str(b) for b in bullets) if isinstance(bullets, list) else str(bullets)
            writer.writerow(row)
    log.info("CSV saved: %s", output_path)


def save_html(slides: list[dict], output_path: Path, video_name: str = "",
             meeting_title: str = "", meeting_date: str = "", meeting_comments: str = "",
             show_image: bool = True, show_bullets: bool = True, show_transcript: bool = True,
             transcript_mode: str = "full", title_slide: dict | None = None,
             recording_speed: float = 1.0) -> None:
    """Write a self-contained HTML report with slide thumbnails and annotations.
    meeting_title/meeting_date/meeting_comments (optional): from the GUI's
    Meeting info section, shown in the report header alongside the video name.
    show_image/show_bullets/show_transcript: toggle which parts of each
    slide card are rendered. transcript_mode: "full" or "first_sentence".
    title_slide (optional): {"image_path", "title", "subtitle"} dict for an
    optional cover page shown before Slide 1. Not counted in "Slides
    detected". See run_pipeline._build_title_slide().
    recording_speed: if not 1.0, every timestamp shown was already converted
    (real_time = video_time / recording_speed); noted in the meta-bar."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cover_html = ""
    if title_slide and title_slide.get("image_path"):
        cover_img = Path(title_slide["image_path"]).name
        cover_html = f"""
<div class="cover-slide">
    <img src="snapshots/{cover_img}" alt="Title slide">
    <div class="cover-text">
        <h2>{title_slide.get('title', '')}</h2>
        <p>{title_slide.get('subtitle', '')}</p>
    </div>
</div>"""

    cards = ""
    for i, slide in enumerate(slides, start=1):
        ts = format_ts(slide.get("timestamp_sec", 0))
        title = slide.get("title") or f"Slide {i}"
        stype = slide.get("slide_type", "")
        bullets = slide.get("bullets", [])
        if isinstance(bullets, str):
            bullets = [bullets]
        transcript = slide.get("transcript_seg", "") or ""
        if transcript_mode == "first_sentence":
            transcript = _first_sentence(transcript)
        snap = slide.get("snapshot_path", "")

        img_tag = ""
        if show_image:
            try:
                snap_rel = Path(snap).name
                img_tag = f'<img src="snapshots/{snap_rel}" alt="Slide {i}" loading="lazy">'
            except Exception:
                img_tag = '<div class="no-img">No snapshot</div>'
        thumb_html = f'<div class="thumb">{img_tag}</div>' if show_image else ""

        bullet_html = ""
        if show_bullets:
            bullet_html = "<ul>" + "".join(f"<li>{b}</li>" for b in bullets[:5]) + "</ul>"
        transcript_html = (
            f'<p class="transcript">{transcript[:300]}</p>'
            if show_transcript and transcript else ""
        )

        cards += f"""
        <div class="card">
            {thumb_html}
            <div class="meta">
                <span class="num">Slide {i}</span>
                <span class="ts">{ts}</span>
                <span class="stype">{stype}</span>
                <h3>{title}</h3>
                {bullet_html}
                {transcript_html}
            </div>
        </div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Slide Report: {video_name}</title>
<style>
  body  {{ font-family: system-ui, sans-serif; margin: 0; padding: 1.5rem 2rem; background: #f5f5f5; color: #222; }}
  h1   {{ font-size: 1.4rem; font-weight: 500; margin-bottom: 0.3rem; }}
  .meta-bar {{ font-size: 0.85rem; color: #666; margin-bottom: 1.5rem; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 1rem; }}
  .card {{ background: #fff; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,.08); display: flex; flex-direction: column; }}
  .thumb img {{ width: 100%; height: 180px; object-fit: cover; display: block; }}
  .no-img {{ height: 180px; background: #e8e8e8; display: flex; align-items: center; justify-content: center; color: #999; font-size: 0.8rem; }}
  .meta {{ padding: 0.75rem 1rem; flex: 1; }}
  .num  {{ font-size: 0.78rem; font-weight: 700; color: #fff; background: #1a56db; padding: 0.15rem 0.5rem; border-radius: 4px; margin-right: 0.3rem; }}
  .ts   {{ font-size: 0.78rem; font-weight: 600; color: #1a56db; background: #eff6ff; padding: 0.15rem 0.45rem; border-radius: 4px; }}
  .stype {{ font-size: 0.75rem; color: #555; background: #f0f0f0; padding: 0.15rem 0.45rem; border-radius: 4px; margin-left: 0.3rem; }}
  h3   {{ font-size: 0.95rem; font-weight: 500; margin: 0.5rem 0 0.3rem; }}
  ul   {{ margin: 0.2rem 0; padding-left: 1.2rem; font-size: 0.83rem; color: #444; }}
  li   {{ margin: 0.1rem 0; }}
  .transcript {{ font-size: 0.78rem; color: #666; margin-top: 0.4rem; border-top: 1px solid #eee; padding-top: 0.4rem; }}
  .meeting-info {{ font-size: 0.85rem; color: #1a56db; margin-bottom: 0.4rem; }}
  .meeting-comments {{ font-size: 0.82rem; color: #666; font-style: italic; margin-bottom: 0.8rem; }}
  .cover-slide {{ background: #fff; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,.08); margin-bottom: 1.5rem; }}
  .cover-slide img {{ width: 100%; max-height: 420px; object-fit: contain; display: block; background: #111; }}
  .cover-text {{ padding: 1rem 1.25rem; text-align: center; }}
  .cover-text h2 {{ font-size: 1.15rem; font-weight: 600; margin: 0 0 0.2rem; }}
  .cover-text p {{ font-size: 0.85rem; color: #666; margin: 0; }}
</style>
</head>
<body>
<h1>{meeting_title or "Slide Report"}</h1>
{f'<div class="meeting-info">{meeting_title + " &nbsp;|&nbsp; " if meeting_title else ""}{meeting_date}</div>' if (meeting_title or meeting_date) else ""}
{f'<div class="meeting-comments">Comments: {meeting_comments}</div>' if meeting_comments else ""}
<div class="meta-bar">
  Video: <strong>{video_name}</strong> &nbsp;|&nbsp;
  Slides detected: <strong>{len(slides)}</strong>
  {f'&nbsp;|&nbsp; Recording speed: <strong>{recording_speed}x</strong> (timestamps converted: real_time = video_time / {recording_speed})' if recording_speed != 1.0 else ""}
</div>
{cover_html}
<div class="grid">{cards}
</div>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    log.info("HTML report saved: %s", output_path)


def save_slide_timing_summary(slides: list[dict], output_path: Path, video_name: str = "",
                              meeting_title: str = "", meeting_date: str = "",
                              meeting_comments: str = "", recording_speed: float = 1.0) -> None:
    """
    Write a plain-text summary listing only the slide number and the
    timestamp it appeared at, one line per slide, no images/bullets/
    transcript. Companion to the full HTML/PDF report for a quick
    "when did slide N change" reference.
    recording_speed: if not 1.0, every timestamp below was already converted
    (real_time = video_time / recording_speed); noted in the header.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"Slide timing summary: {video_name}"]
    if meeting_title:
        lines.append(f"Meeting: {meeting_title}")
    if meeting_date:
        lines.append(f"Date: {meeting_date}")
    if meeting_comments:
        lines.append(f"Comments: {meeting_comments}")
    if recording_speed != 1.0:
        lines.append(f"Recording speed: {recording_speed}x (timestamps converted: "
                     f"real_time = video_time / {recording_speed})")
    lines += [f"Slides detected: {len(slides)}", ""]
    for i, slide in enumerate(slides, start=1):
        ts = format_ts(slide.get("timestamp_sec", 0))
        title = slide.get("title") or ""
        line = f"Slide {i:03d}  {ts}"
        if title:
            line += f"  {title}"
        lines.append(line)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    log.info("Slide timing summary saved: %s", output_path)


class _JSONEncoder(json.JSONEncoder):
    """Handles numpy scalar types (int64, float32, etc.) from whisperx/pandas."""
    def default(self, obj):
        try:
            import numpy as np
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
        except ImportError:
            pass
        return super().default(obj)


def save_json(slides: list[dict], output_path: Path) -> None:
    """Write full slide index as JSON for downstream processing."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(slides, fh, indent=2, ensure_ascii=False, cls=_JSONEncoder)
    log.info("JSON index saved: %s", output_path)


def save_html_for_pdf(slides: list[dict], output_path: Path, video_name: str = "",
                      meeting_title: str = "", meeting_date: str = "",
                      meeting_comments: str = "", show_image: bool = True,
                      show_bullets: bool = True, show_transcript: bool = True,
                      transcript_mode: str = "full", title_slide: dict | None = None,
                      recording_speed: float = 1.0) -> None:
    """
    Generate a PDF-optimised HTML file with exactly one slide per page.
    Used exclusively as input to save_pdf_from_html, not for browser viewing.
    meeting_title/meeting_date/meeting_comments (optional): from the GUI's
    Meeting info section, shown on the cover block.
    show_image/show_bullets/show_transcript: toggle which parts of each
    slide page are rendered. transcript_mode: "full" or "first_sentence"
    (first sentence only, so the segment reliably fits on one page).
    title_slide (optional): {"image_path", "title", "subtitle"} dict for an
    optional full-page cover image shown as its own page before the meta
    cover block / Slide 1. See run_pipeline._build_title_slide().
    recording_speed: if not 1.0, every timestamp shown was already converted
    (real_time = video_time / recording_speed); noted on the cover block.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cover_slide_page = ""
    if title_slide and title_slide.get("image_path"):
        cover_img = Path(title_slide["image_path"]).name
        cover_slide_page = f"""
<div class="slide-page cover-image-page" style="page-break-after: always;">
  <div class="cover-image"><img src="snapshots/{cover_img}" alt="Title slide"></div>
  <div class="cover-caption">
    <h1>{title_slide.get('title', '')}</h1>
    <p>{title_slide.get('subtitle', '')}</p>
  </div>
</div>"""

    pages = ""
    for i, slide in enumerate(slides, start=1):
        ts      = format_ts(slide.get("timestamp_sec", 0))
        title   = slide.get("title") or f"Slide {i}"
        stype   = slide.get("slide_type", "")
        bullets = slide.get("bullets", [])
        if isinstance(bullets, str):
            bullets = [bullets]
        transcript = slide.get("transcript_seg", "") or ""
        transcript = _first_sentence(transcript) if transcript_mode == "first_sentence" else transcript[:400]
        snap = slide.get("snapshot_path", "")

        img_html = ""
        if show_image:
            try:
                snap_rel = Path(snap).name
                img_html = f'<img src="snapshots/{snap_rel}" alt="Slide {i}">'
            except Exception:
                img_html = '<div class="no-snap">No snapshot</div>'
        snap_html = f'<div class="snap">{img_html}</div>' if show_image else ""

        bullet_html = ("<ul>" + "".join(f"<li>{b}</li>" for b in bullets[:5]) + "</ul>") if show_bullets else ""
        transcript_html = (
            f'<div class="transcript"><strong>Transcript:</strong> {transcript}</div>'
            if show_transcript and transcript else ""
        )
        page_break = "page-break-after: always;" if i < len(slides) else ""

        pages += f"""
<div class="slide-page" style="{page_break}">
  <div class="slide-header">
    <span class="ts">{ts}</span>
    <span class="stype">{stype}</span>
    <span class="num">Slide {i} / {len(slides)}</span>
  </div>
  {snap_html}
  <div class="annotation">
    <h2>{title}</h2>
    {bullet_html}
    {transcript_html}
  </div>
</div>"""

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Slide Report: {video_name}</title>
<style>
  @page {{ size: A4 landscape; margin: 12mm 14mm; }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: Arial, Helvetica, sans-serif; font-size: 10pt; color: #1a1a1a; background: #fff; }}
  .slide-page {{ width: 100%; height: 185mm; display: flex; flex-direction: column; gap: 4mm; }}
  .slide-header {{ display: flex; align-items: center; gap: 6px; font-size: 8pt; color: #555; border-bottom: 1px solid #ddd; padding-bottom: 2mm; }}
  .ts {{ font-weight: bold; color: #1a56db; background: #eff6ff; padding: 1px 6px; border-radius: 3px; }}
  .stype {{ background: #f0f0f0; padding: 1px 6px; border-radius: 3px; }}
  .num {{ margin-left: auto; color: #999; }}
  .snap {{ flex: 1; overflow: hidden; text-align: center; background: #f8f8f8; border: 1px solid #e0e0e0; border-radius: 4px; }}
  .snap img {{ max-width: 100%; max-height: 100%; object-fit: contain; display: block; margin: auto; }}
  .no-snap {{ height: 80mm; display: flex; align-items: center; justify-content: center; color: #bbb; font-size: 9pt; }}
  .annotation {{ min-height: 30mm; padding: 3mm 0 0; }}
  .annotation h2 {{ font-size: 11pt; font-weight: bold; color: #1a1a1a; margin-bottom: 2mm; }}
  ul {{ padding-left: 5mm; font-size: 9pt; color: #333; line-height: 1.5; }}
  .transcript {{ font-size: 8pt; color: #666; margin-top: 2mm; border-top: 1px solid #eee; padding-top: 2mm; line-height: 1.4; }}
  .cover {{ font-size: 9pt; color: #888; margin-bottom: 4mm; border-bottom: 2px solid #1a56db; padding-bottom: 3mm; }}
  .cover strong {{ color: #1a56db; font-size: 13pt; }}
  .cover .comments {{ font-style: italic; color: #666; margin-top: 1mm; }}
  .cover-image-page {{ align-items: center; justify-content: center; }}
  .cover-image {{ flex: 1; display: flex; align-items: center; justify-content: center; overflow: hidden; }}
  .cover-image img {{ max-width: 100%; max-height: 100%; object-fit: contain; }}
  .cover-caption {{ text-align: center; padding: 4mm 0; }}
  .cover-caption h1 {{ font-size: 16pt; color: #1a1a1a; margin-bottom: 1mm; }}
  .cover-caption p {{ font-size: 10pt; color: #666; }}
</style>
</head>
<body>
{cover_slide_page}
<div class="cover">
  <strong>{meeting_title or ("Slide Report: " + video_name)}</strong><br>
  {f"Video: {video_name}<br>" if meeting_title else ""}
  {f"Date: {meeting_date}<br>" if meeting_date else ""}
  Slides detected: {len(slides)}
  {f'<br>Recording speed: {recording_speed}x (timestamps converted: real_time = video_time / {recording_speed})' if recording_speed != 1.0 else ""}
  {f'<div class="comments">Comments: {meeting_comments}</div>' if meeting_comments else ""}
</div>
{pages}
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(html)
    log.info("PDF-optimised HTML saved: %s", output_path)


# ---------------------------------------------------------------------------
# PDF export: Playwright/Chromium (preferred) -> weasyprint -> pdfkit
#
# Rationale (2026-07-02): weasyprint depends on the GTK3 native library
# stack on Windows. On machines that also have Tesseract-OCR installed,
# Tesseract ships its own same-named libgobject-2.0-0.dll earlier in PATH,
# which weasyprint's ctypes loader picks up instead of the real GTK3
# runtime, failing with "cannot load library ...: error 0x7e". Playwright's
# headless Chromium has no such native dependency conflict, so it is tried
# first. weasyprint/pdfkit remain as fallbacks for machines without
# Playwright/Chromium installed.
# ---------------------------------------------------------------------------

def _save_pdf_playwright(html_path: Path, pdf_path: Path) -> bool:
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(html_path.resolve().as_uri())
        page.pdf(path=str(pdf_path), print_background=True, format="A4")
        browser.close()
    return True


def _save_pdf_weasyprint(html_path: Path, pdf_path: Path) -> bool:
    from weasyprint import HTML as WeasyprintHTML
    WeasyprintHTML(filename=str(html_path)).write_pdf(str(pdf_path))
    return True


def _save_pdf_pdfkit(html_path: Path, pdf_path: Path) -> bool:
    import pdfkit
    pdfkit.from_file(str(html_path), str(pdf_path))
    return True


def save_pdf_from_html(html_path: Path, pdf_path: Path) -> bool:
    """
    Convert an HTML file to PDF. Tries Playwright/Chromium first (most
    reliable on Windows, no native library conflicts), then weasyprint,
    then pdfkit. Returns True on success.
    """
    html_path = Path(html_path)
    pdf_path = Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    backends = [
        ("Playwright/Chromium", _save_pdf_playwright),
        ("weasyprint", _save_pdf_weasyprint),
        ("pdfkit", _save_pdf_pdfkit),
    ]

    for name, fn in backends:
        try:
            fn(html_path, pdf_path)
            log.info("PDF saved via %s: %s", name, pdf_path)
            return True
        except ImportError:
            log.debug("%s not installed; trying next PDF backend.", name)
        except Exception as exc:
            log.warning("%s failed: %s; trying next PDF backend.", name, exc)

    log.error(
        "No working PDF backend found. Install one: "
        "'pip install playwright && playwright install chromium' (recommended), "
        "'pip install weasyprint' (needs GTK3 runtime on Windows), or "
        "'pip install pdfkit' (needs wkhtmltopdf binary in PATH)."
    )
    return False
