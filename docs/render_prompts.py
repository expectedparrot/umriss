from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


def text(value: Any) -> str:
    if value in (None, ""):
        return "—"
    return html.escape(str(value))


def render_record(row: dict[str, Any], number: int) -> str:
    pattern = row.get("pattern") or {}
    pattern_rows = "".join(
        f"<tr><td>{text(item)}</td><td>{text(answer)}</td></tr>"
        for item, answer in pattern.items()
    )
    if not pattern_rows:
        pattern_rows = (
            f"<tr><td>{text(row.get('coverage_item'))}</td>"
            f"<td>{text(row.get('coverage_option'))}</td></tr>"
        )

    coverage = ""
    if row.get("coverage_item"):
        coverage = (
            "<p><strong>Coverage target:</strong> "
            f"{text(row.get('coverage_item'))} → {text(row.get('coverage_option'))}. "
            f"<strong>Style:</strong> {text(row.get('coverage_style'))}.</p>"
        )

    return f"""
    <section id="support-{number}">
      <h2>{number}. {text(row.get("variant", "support point"))}</h2>
      <table class="metadata"><tbody>
        <tr><th>Job ID</th><td><code>{text(row.get("job_id"))}</code></td></tr>
        <tr><th>Support ID</th><td>{text(row.get("support_id"))}</td></tr>
        <tr><th>Battery</th><td>{text(row.get("battery"))}</td></tr>
        <tr><th>Moment dimension</th><td>{text(row.get("moment_dimension"))}</td></tr>
      </tbody></table>
      {coverage}
      <h3>Intended response scaffold</h3>
      <table><thead><tr><th>Item</th><th>Target region</th></tr></thead>
        <tbody>{pattern_rows}</tbody>
      </table>
      <details>
        <summary>Show complete model prompt</summary>
        <pre><code>{text(row.get("prompt"))}</code></pre>
      </details>
    </section>
    """


def render(source: Path, output: Path) -> None:
    rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    nav = "".join(
        f'<a href="#support-{index}">{index}. {text(row.get("variant", "support point"))}</a>'
        for index, row in enumerate(rows, 1)
    )
    sections = "".join(render_record(row, index) for index, row in enumerate(rows, 1))
    document = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>umriss | Rendered support prompts</title>
  <style>
    :root{{--green:#428a5f;--dark:#214d35;--ink:#17201a;--muted:#667069;--paper:#f5f7f5;--rule:#dfe5e0;--code:#19231d;--sidebar:260px;--serif:Georgia,"Times New Roman",serif;--sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif;--mono:"SFMono-Regular",Consolas,monospace}}
    *{{box-sizing:border-box}}html{{scroll-behavior:smooth}}body{{margin:0;color:var(--ink);background:var(--paper);font:16px/1.65 var(--sans)}}
    nav{{position:fixed;inset:0 auto 0 0;width:var(--sidebar);overflow:auto;padding:26px 20px;background:var(--dark);color:#fff}}
    nav strong{{display:block;margin-bottom:4px;font:22px var(--serif)}}nav p{{margin:0 0 22px;color:#bdd3c5;font-size:12px}}
    nav a{{display:block;padding:6px 8px;color:#d5e9dc;text-decoration:none;border-radius:5px;font-size:13px}}nav a:hover{{color:#fff;background:#2d6244}}
    main{{max-width:920px;margin-left:var(--sidebar);padding:52px 52px 100px;background:#fff}}header{{padding-bottom:34px;border-bottom:3px solid var(--green)}}
    h1,h2,h3{{font-family:var(--serif);font-weight:600}}h1{{margin:0 0 12px;font-size:44px}}h2{{margin:70px 0 20px;padding-top:14px;color:var(--green);border-top:1px solid var(--rule)}}h3{{margin-top:28px}}
    .lede{{max-width:740px;color:var(--muted);font-size:19px}}table{{width:100%;max-width:760px;border-collapse:collapse}}th,td{{padding:9px 11px;text-align:left;border-bottom:1px solid var(--rule)}}thead th{{color:#fff;background:var(--green)}}.metadata th{{width:180px;color:var(--dark);background:#f1f5f2}}
    code{{font-family:var(--mono)}}details{{max-width:760px;margin:22px 0}}summary{{color:var(--green);font-weight:700;cursor:pointer}}pre{{overflow-x:hidden;padding:20px;color:#d8e3db;background:var(--code);border-radius:7px;white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.55 var(--mono)}}
    @media(max-width:850px){{nav{{position:static;width:auto}}nav a{{display:inline-block}}main{{margin:0;padding:32px 24px 80px}}}}
  </style>
</head>
<body>
<nav><strong>umriss prompts</strong><p>{len(rows)} support points · {text(rows[0].get("battery") if rows else "")}</p>{nav}</nav>
<main>
  <header><h1>Rendered support prompts</h1>
    <p class="lede">A readable view of <code>{text(source.name)}</code>. The JSONL remains the canonical machine-readable artifact; this page presents the same records for review.</p>
  </header>
  {sections}
</main>
</body>
</html>
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render an umriss prompt JSONL file as HTML.")
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    render(args.source, args.output)


if __name__ == "__main__":
    main()
