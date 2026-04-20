"""
display.py — HTML/CSS rendering for Colab and IPython notebooks.

Falls back to plain JSON print when IPython is unavailable (scripts, CLI).

Public API:
    render_ledger_table(entries, project_name="")
    render_parse_result(summary)
"""

from __future__ import annotations

import json
from typing import Dict, List

# ── CSS ───────────────────────────────────────────────────────────────────────

_CSS = """
<style>
.adsl-wrap { font-family: 'Segoe UI', Arial, sans-serif; margin: 8px 0; }
.adsl-title { font-size: 14px; font-weight: 600; color: #555; margin-bottom: 6px; }
.adsl-table { border-collapse: collapse; width: 100%; font-size: 12px; }
.adsl-table th { background: #1e1e2e; color: #cdd6f4; padding: 6px 10px;
                  text-align: left; white-space: nowrap; }
.adsl-table td { border-bottom: 1px solid #e8e8e8; padding: 5px 10px;
                  vertical-align: top; max-width: 300px; }
.adsl-table tr:hover td { background: #f8f8ff; }
.adsl-tag { display: inline-block; border-radius: 3px; padding: 1px 6px;
             margin: 1px; font-size: 11px; white-space: nowrap; }
.adsl-tag.model { background: #e3f2fd; color: #0d47a1; }
.adsl-tag.metric { background: #e8f5e9; color: #1b5e20; }
.adsl-tag.prep { background: #fff3e0; color: #bf360c; }
.adsl-tag.empty { background: #f5f5f5; color: #9e9e9e; font-style: italic; }
.adsl-card { border: 1px solid #e0e0e0; border-radius: 6px; padding: 12px 16px;
              background: #fafafa; margin: 8px 0; }
.adsl-card-title { font-weight: 600; font-size: 13px; margin-bottom: 8px; color: #333; }
.adsl-row { margin: 4px 0; }
.adsl-label { font-size: 11px; color: #888; margin-right: 6px; }
</style>
"""


# ── Public API ────────────────────────────────────────────────────────────────

def render_ledger_table(entries: List[Dict], project_name: str = "") -> None:
    """
    Render entries as an HTML table in IPython/Colab.
    Falls back to plain print if IPython is unavailable.
    """
    if not entries:
        _display_or_print(
            f'{_CSS}<div class="adsl-wrap"><em>No entries yet. Call ledger.parse() to add files.</em></div>',
            [],
        )
        return

    title = f"DS Ledger — {project_name}" if project_name else "DS Ledger"
    rows = "".join(_table_row(e) for e in entries)
    html = f"""{_CSS}
<div class="adsl-wrap">
  <div class="adsl-title">📋 {title} ({len(entries)} entries)</div>
  <table class="adsl-table">
    <thead>
      <tr>
        <th>#</th>
        <th>File</th>
        <th>Type</th>
        <th>Models</th>
        <th>Metrics</th>
        <th>Preprocessing</th>
        <th>Timestamp</th>
      </tr>
    </thead>
    <tbody>{rows}</tbody>
  </table>
</div>"""
    _display_or_print(html, entries)


def render_parse_result(summary: Dict) -> None:
    """Render a compact card for a single parse result (shown after ledger.parse())."""
    models = _tag_list(summary.get("models", []), "model")
    metrics = _tag_list(summary.get("metrics", []), "metric")
    preps = _tag_list(summary.get("preprocessing", []), "prep")

    entry_id = summary.get("entry_id", "?")
    file_name = summary.get("file", "")

    html = f"""{_CSS}
<div class="adsl-card">
  <div class="adsl-card-title">✅ Parsed: {file_name} — entry #{entry_id}</div>
  <div class="adsl-row"><span class="adsl-label">Models</span>{models}</div>
  <div class="adsl-row"><span class="adsl-label">Metrics</span>{metrics}</div>
  <div class="adsl-row"><span class="adsl-label">Preprocessing</span>{preps}</div>
</div>"""
    _display_or_print(html, summary)


# ── Private helpers ────────────────────────────────────────────────────────────

def _table_row(entry: Dict) -> str:
    ts = (entry.get("timestamp") or "")[:16]
    file_name = entry.get("file_path", "")
    file_type = entry.get("file_type", "?")
    models = _tag_list(entry.get("models", []), "model")
    metrics = _tag_list(entry.get("metrics", []), "metric")
    preps = _tag_list(entry.get("preprocessing", []), "prep")
    return (
        f"<tr>"
        f"<td>{entry.get('id', '')}</td>"
        f"<td style='max-width:180px;overflow:hidden;text-overflow:ellipsis' title='{file_name}'>{file_name}</td>"
        f"<td>{file_type}</td>"
        f"<td>{models}</td>"
        f"<td>{metrics}</td>"
        f"<td>{preps}</td>"
        f"<td style='white-space:nowrap'>{ts}</td>"
        f"</tr>"
    )


def _tag_list(items, css_class: str) -> str:
    if not items:
        return '<span class="adsl-tag empty">—</span>'
    tags = []
    for item in items:
        if isinstance(item, dict):
            name = item.get("name", str(item))
        else:
            name = str(item)
        tags.append(f'<span class="adsl-tag {css_class}">{name}</span>')
    return "".join(tags)


def _display_or_print(html: str, fallback) -> None:
    try:
        from IPython.display import display, HTML
        display(HTML(html))
    except ImportError:
        # Not in IPython — print plain text
        if isinstance(fallback, list):
            for item in fallback:
                print(json.dumps({k: v for k, v in item.items()
                                  if k not in ("environment", "raw_text")},
                                 default=str, indent=2))
        else:
            print(json.dumps(fallback, default=str, indent=2))
