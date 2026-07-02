# Prompt templates

Every `.md` file in this folder is a selectable notes/summary template.
It shows up automatically in the GUI dropdown (Run tab) and via
`--prompt-template <file>` on the command line, no code changes needed
to add a new one.

## How a template is used

1. The whole file content is sent to the LLM as the system prompt.
2. The pipeline looks for the first line starting with `## ` in the file
   and treats it as the required first heading of the output. This is
   used both to instruct the model ("your response MUST begin with
   exactly ...") and as the assistant-prefill for the Ollama backend,
   which keeps long-context local models from adding a preamble.
3. The transcript is appended as a user message and, for long
   transcripts, processed with map-reduce chunking (see `notes.py`).
4. The same parsed structure (`## HEADING`, `- bullet`, `"quote"`) drives
   all four notes output formats: txt, html, pdf, and docx - so a new
   template automatically renders correctly in every format without any
   extra work.

## Writing a new template

Copy `meeting.md` or `webinar.md` as a starting point and adjust the
headings and rules. Keep these conventions:

- Start the instructions with the assistant's role and the fact-only rule.
- Use `## SECTION NAME` headings only (no `#`, no `###`).
- Use flat `- ` bullet lists only (no nested lists, no tables) so the
  output renders correctly in OneNote/Word and in the docx export.
- Give a placeholder line for empty sections (e.g. `"No action items identified"`)
  so the model does not omit a heading it has nothing to say under.

## Shipped templates

| File | Purpose |
|---|---|
| `meeting.md`  | Internal meeting notes: participants, decisions, action items |
| `webinar.md`  | Webinar/conference summary: topics, takeaways, quotes, Q&A |
