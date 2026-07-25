from __future__ import annotations

import html
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "examples" / "pew_w154" / "run" / "derived"
DETAIL = DERIVED / "pew_w154_diff1_uniform_n208_generated_support_detail.csv"
SUMMARY = DERIVED / "pew_w154_diff1_uniform_n208_generated_support_summary.csv"
OUT = ROOT / "docs" / "assets"

COLORS = {
    "generated support mixture": "#2f7d57",
    "unconditioned one-shot": "#d97706",
    "structured two-step": "#8b5cf6",
    "unweighted support bank": "#64748b",
    "uniform": "#cbd5e1",
}
LABELS = {
    "generated support mixture": "Marginally weighted twins",
    "unconditioned one-shot": "Direct one-shot",
    "structured two-step": "Structured two-step",
    "unweighted support bank": "Unweighted support bank",
    "uniform": "Uniform",
}


def write_svg(path: Path, body: str, width: int, height: int, title: str) -> None:
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">'
        f'<title id="title">{html.escape(title)}</title>'
        '<desc id="desc">Bars are generated directly from checked-in leave-one-out result CSV files.</desc>'
        '<style>text{font-family:system-ui,-apple-system,sans-serif;fill:#17211c}'
        '.label{font-size:14px}.small{font-size:12px;fill:#5d6962}.value{font-size:12px;font-weight:650}'
        '.axis{stroke:#cbd5cf;stroke-width:1}</style>'
        f"{body}</svg>\n"
    )


def summary_chart() -> None:
    frame = pd.read_csv(SUMMARY).set_index("method")
    methods = [
        "unconditioned one-shot",
        "structured two-step",
        "generated support mixture",
        "unweighted support bank",
        "uniform",
    ]
    panels = [("mean_rmse", "Mean RMSE", 0.25), ("mean_kl_divergence", "Mean KL divergence (nats)", 0.16)]
    body = []
    for panel_idx, (column, title, maximum) in enumerate(panels):
        x0 = 205 + panel_idx * 425
        body.append(f'<text x="{x0}" y="30" class="label" font-weight="700">{title}</text>')
        body.append(f'<line x1="{x0}" y1="45" x2="{x0 + 300}" y2="45" class="axis"/>')
        for tick in range(6):
            x = x0 + tick * 60
            value = maximum * tick / 5
            body.append(f'<line x1="{x}" y1="42" x2="{x}" y2="48" class="axis"/>')
            body.append(f'<text x="{x}" y="62" text-anchor="middle" class="small">{value:.2f}</text>')
        for idx, method in enumerate(methods):
            y = 85 + idx * 48
            if panel_idx == 0:
                body.append(f'<text x="195" y="{y + 14}" text-anchor="end" class="label">{LABELS[method]}</text>')
            value = float(frame.loc[method, column])
            width = 300 * value / maximum
            body.append(
                f'<rect x="{x0}" y="{y}" width="{width:.1f}" height="20" rx="3" fill="{COLORS[method]}"/>'
            )
            body.append(f'<text x="{x0 + width + 6:.1f}" y="{y + 15}" class="value">{value:.3f}</text>')
    write_svg(OUT / "loo-summary.svg", "".join(body), 1020, 345, "Mean held-out error by method")


def item_chart() -> None:
    detail = pd.read_csv(DETAIL)
    methods = ["generated support mixture", "unconditioned one-shot", "uniform"]
    items = list(detail["holdout"].drop_duplicates())
    item_labels = {
        "hobbies": "Hobbies",
        "physical_abilities": "Physical abilities",
        "parenting": "Parenting",
        "feelings": "Expression of feelings",
        "workplace": "Workplace abilities",
    }
    x0, chart_width, maximum = 205, 620, 0.42
    body = [
        '<text x="205" y="30" class="label" font-weight="700">RMSE on each completely omitted marginal</text>',
        f'<line x1="{x0}" y1="48" x2="{x0 + chart_width}" y2="48" class="axis"/>',
    ]
    for tick in range(8):
        x = x0 + tick * chart_width / 7
        value = maximum * tick / 7
        body.append(f'<line x1="{x:.1f}" y1="45" x2="{x:.1f}" y2="51" class="axis"/>')
        body.append(f'<text x="{x:.1f}" y="65" text-anchor="middle" class="small">{value:.2f}</text>')
    for item_idx, item in enumerate(items):
        base_y = 88 + item_idx * 83
        body.append(f'<text x="195" y="{base_y + 28}" text-anchor="end" class="label">{item_labels[item]}</text>')
        for method_idx, method in enumerate(methods):
            row = detail[(detail["holdout"] == item) & (detail["method"] == method)].iloc[0]
            value = float(row["rmse"])
            y = base_y + method_idx * 22
            width = chart_width * value / maximum
            body.append(
                f'<rect x="{x0}" y="{y}" width="{width:.1f}" height="16" rx="2" fill="{COLORS[method]}"/>'
            )
            body.append(f'<text x="{x0 + width + 5:.1f}" y="{y + 13}" class="value">{value:.3f}</text>')
    legend_x = [210, 430, 620]
    for x, method in zip(legend_x, methods, strict=True):
        body.append(f'<rect x="{x}" y="510" width="14" height="14" rx="2" fill="{COLORS[method]}"/>')
        body.append(f'<text x="{x + 20}" y="522" class="small">{LABELS[method]}</text>')
    write_svg(OUT / "loo-by-item.svg", "".join(body), 920, 550, "Held-out RMSE by survey item")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary_chart()
    item_chart()


if __name__ == "__main__":
    main()
