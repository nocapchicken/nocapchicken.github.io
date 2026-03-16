# AI-assisted (Claude Code, claude.ai) — https://claude.ai
"""Parse REQUIREMENTS_CHECKLIST.md and write docs/checklist/index.html."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parent.parent
SOURCE = ROOT / "REQUIREMENTS_CHECKLIST.md"
OUT = ROOT / "docs" / "checklist" / "index.html"


def parse_checklist(text: str) -> list[dict]:
    """Parse REQUIREMENTS_CHECKLIST.md into a list of section dicts with item completion state."""
    sections: list[dict] = []
    current: dict | None = None

    for line in text.splitlines():
        # Section heading
        m = re.match(r"^## (.+)", line)
        if m:
            current = {"title": m.group(1).strip(), "items": []}
            sections.append(current)
            continue

        if current is None:
            continue

        # Table row:  | ID | description | anything | ✅/⬜ |
        m = re.match(
            r"^\|\s*([A-Z]+\d+)\s*\|(.+?)\|.*?\|\s*(✅|⬜)\s*\|", line
        )
        if m:
            item_id, label, status = m.groups()
            current["items"].append(
                {"id": item_id.strip(), "label": _clean(label), "done": status == "✅"}
            )
            continue

        # List item:  - [x] **ID** — description
        m = re.match(r"^-\s+\[(x| )\]\s+\*\*([A-Z]+\d+)\*\*\s*[—–-]?\s*(.+)", line)
        if m:
            checked, item_id, label = m.groups()
            current["items"].append(
                {"id": item_id.strip(), "label": _clean(label), "done": checked == "x"}
            )

    return [s for s in sections if s["items"]]


def _clean(text: str) -> str:
    """Strip markdown formatting for plain display."""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"`(.+?)`", r"\1", text)
    return text.strip(" —–-")


def _progress(items: list[dict]) -> tuple[int, int]:
    done = sum(1 for i in items if i["done"])
    return done, len(items)


def render_html(sections: list[dict], generated_at: str) -> str:
    """Render parsed checklist sections into a self-contained HTML string."""
    total_done = sum(i["done"] for s in sections for i in s["items"])
    total_all = sum(len(s["items"]) for s in sections)
    total_pct = round(100 * total_done / total_all) if total_all else 0

    section_html = ""
    for sec in sections:
        done, total = _progress(sec["items"])
        pct = round(100 * done / total) if total else 0

        rows = ""
        for item in sec["items"]:
            status_cls = "done" if item["done"] else "open"
            glyph = "✓" if item["done"] else "○"
            rows += (
                f'<tr class="{status_cls}">'
                f'<td class="item-glyph">{glyph}</td>'
                f'<td class="item-id">{item["id"]}</td>'
                f'<td class="item-label">{item["label"]}</td>'
                f"</tr>\n"
            )

        section_html += f"""
    <section class="checklist-section">
      <div class="section-header">
        <h2>{sec['title']}</h2>
        <span class="section-count">{done}/{total}</span>
      </div>
      <div class="progress-bar"><div class="progress-fill" style="width:{pct}%"></div></div>
      <table class="item-table">
        <tbody>{rows}</tbody>
      </table>
    </section>
"""

    return f"""<!-- AI-assisted (Claude Code, claude.ai) — https://claude.ai -->
<!-- AUTO-GENERATED — edit REQUIREMENTS_CHECKLIST.md, not this file -->
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Requirements Checklist — nocapchicken</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet" />
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    :root {{
      --bg: #f8f7f4; --surface: #ffffff; --border: #e5e2db;
      --text: #1a1916; --text-muted: #6b6860;
      --green: #16a34a; --green-bg: #f0fdf4;
      --radius-sm: 6px; --radius-md: 12px;
      --shadow-sm: 0 1px 3px rgba(0,0,0,.07);
      --font: 'Inter', system-ui, -apple-system, sans-serif;
    }}
    html {{ font-size: 16px; -webkit-font-smoothing: antialiased; }}
    body {{ font-family: var(--font); background: var(--bg); color: var(--text); min-height: 100dvh; }}
    a {{ color: inherit; }}
    .container {{ width: 100%; max-width: 780px; margin: 0 auto; padding: 0 24px; }}

    /* header */
    .site-header {{ border-bottom: 1px solid var(--border); padding: 20px 0; background: var(--surface); }}
    .site-header .container {{ display: flex; align-items: center; gap: 16px; }}
    .back-link {{ font-size: 0.85rem; color: var(--text-muted); text-decoration: none; }}
    .back-link:hover {{ color: var(--text); }}
    .header-title {{ font-size: 1rem; font-weight: 600; }}

    /* overall progress */
    .overall {{ background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-md);
      padding: 24px; margin: 32px 0 24px; box-shadow: var(--shadow-sm); }}
    .overall-row {{ display: flex; align-items: baseline; gap: 12px; margin-bottom: 12px; }}
    .overall-pct {{ font-size: 2rem; font-weight: 700; }}
    .overall-label {{ font-size: 0.9rem; color: var(--text-muted); }}
    .progress-bar {{ height: 8px; background: var(--border); border-radius: 4px; overflow: hidden; margin-top: 4px; }}
    .progress-fill {{ height: 100%; background: var(--green); border-radius: 4px; transition: width .3s; }}

    /* sections */
    .checklist-section {{ background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius-md);
      padding: 20px 24px; margin-bottom: 16px; box-shadow: var(--shadow-sm); }}
    .section-header {{ display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 8px; }}
    .section-header h2 {{ font-size: 0.95rem; font-weight: 600; }}
    .section-count {{ font-size: 0.8rem; color: var(--text-muted); font-variant-numeric: tabular-nums; }}
    .checklist-section .progress-bar {{ margin-bottom: 16px; }}

    /* item table */
    .item-table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
    .item-table tr {{ border-top: 1px solid var(--border); }}
    .item-table tr:first-child {{ border-top: none; }}
    .item-table td {{ padding: 7px 4px; vertical-align: top; }}
    .item-glyph {{ width: 20px; color: var(--text-muted); font-size: 0.75rem; padding-top: 8px; }}
    .item-id {{ width: 52px; font-weight: 600; font-size: 0.78rem; white-space: nowrap; padding-top: 8px; }}
    .item-label {{ color: var(--text-muted); line-height: 1.45; }}
    tr.done .item-glyph {{ color: var(--green); }}
    tr.done .item-id {{ color: var(--text); }}
    tr.done .item-label {{ color: var(--text); }}

    /* footer */
    .page-footer {{ text-align: center; font-size: 0.8rem; color: var(--text-muted); padding: 32px 0 48px; }}
  </style>
</head>
<body>
  <header class="site-header">
    <div class="container">
      <a class="back-link" href="../">← nocapchicken</a>
      <span class="header-title">Requirements Checklist</span>
    </div>
  </header>

  <main style="padding-bottom: 48px;">
    <div class="container">
      <div class="overall">
        <div class="overall-row">
          <span class="overall-pct">{total_pct}%</span>
          <span class="overall-label">{total_done} of {total_all} items complete</span>
        </div>
        <div class="progress-bar"><div class="progress-fill" style="width:{total_pct}%"></div></div>
      </div>
      {section_html}
    </div>
  </main>

  <footer class="page-footer">
    Auto-generated from <code>REQUIREMENTS_CHECKLIST.md</code> · {generated_at}
  </footer>
</body>
</html>
"""


def main() -> None:
    """Read REQUIREMENTS_CHECKLIST.md and write the rendered HTML to docs/checklist/index.html."""
    text = SOURCE.read_text()
    sections = parse_checklist(text)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    html = render_html(sections, generated_at)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html)
    print(f"Written → {OUT}")


if __name__ == "__main__":
    main()
