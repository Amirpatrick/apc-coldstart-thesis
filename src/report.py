"""
Reporting helpers.

Every check writes three kinds of artefact into the repo (not the workspace),
so that the whole evidence trail is versioned and shareable:

    reports/check_N_<name>.md      human-readable finding + verdict
    reports/tables/*.csv           the numbers behind it
    reports/figures/*.png          the plot

The `Report` object accumulates sections and writes once at the end.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe; figures are saved, not shown
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from . import config as C  # noqa: E402


class Report:
    """Accumulates markdown sections, tables and figures for one check."""

    def __init__(self, number: int, slug: str, title: str, question: str):
        C.ensure_dirs()
        self.number = number
        self.slug = slug
        self.title = title
        self.question = question
        self._parts: list[str] = []
        self._verdict: str | None = None

    # -- content -----------------------------------------------------------

    def text(self, s: str) -> "Report":
        self._parts.append(s.strip() + "\n")
        return self

    def heading(self, s: str, level: int = 2) -> "Report":
        self._parts.append(f"\n{'#' * level} {s}\n")
        return self

    def kv(self, mapping: dict) -> "Report":
        lines = ["| Quantity | Value |", "|---|---|"]
        for k, v in mapping.items():
            lines.append(f"| {k} | {v} |")
        self._parts.append("\n".join(lines) + "\n")
        return self

    def table(self, df: pd.DataFrame, name: str, max_rows: int = 40) -> "Report":
        """Write the full table to CSV, embed a (possibly truncated) preview."""
        path = C.TABLE_DIR / f"check{self.number}_{name}.csv"
        df.to_csv(path, index=False)
        shown = df.head(max_rows)
        self._parts.append(shown.to_markdown(index=False) + "\n")
        if len(df) > max_rows:
            self._parts.append(f"\n*(showing {max_rows} of {len(df):,} rows)*\n")
        self._parts.append(f"\nFull table: `reports/tables/{path.name}`\n")
        return self

    def figure(self, fig, name: str, caption: str = "") -> "Report":
        path = C.FIG_DIR / f"check{self.number}_{name}.png"
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        rel = f"figures/{path.name}"
        self._parts.append(f"\n![{caption or name}]({rel})\n")
        if caption:
            self._parts.append(f"\n*{caption}*\n")
        return self

    def verdict(self, decision: str, rationale: str) -> "Report":
        """The go/no-go call this check exists to make."""
        self._verdict = f"**{decision}**\n\n{rationale.strip()}"
        return self

    # -- output ------------------------------------------------------------

    def write(self) -> Path:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        header = [
            f"# Check {self.number} — {self.title}",
            "",
            f"**Question this check answers.** {self.question}",
            "",
            f"*Generated {stamp} · seed {C.SEED} · "
            f"warm threshold = {C.WARM_MIN_APPEARANCES} playlist appearances*",
            "",
            "---",
            "",
        ]
        body = "".join(self._parts)
        footer = ""
        if self._verdict:
            footer = "\n---\n\n## Verdict\n\n" + self._verdict + "\n"

        path = C.REPORT_DIR / f"check{self.number}_{self.slug}.md"
        path.write_text("\n".join(header) + body + footer, encoding="utf-8")
        print(f"  -> wrote {path}")
        return path


def describe_series(s: pd.Series, name: str) -> pd.DataFrame:
    """Standard distribution summary used across several checks."""
    q = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    rows = {
        "count": len(s),
        "mean": s.mean(),
        "std": s.std(),
        "min": s.min(),
    }
    for p in q:
        rows[f"p{int(p * 100)}"] = s.quantile(p)
    rows["max"] = s.max()
    return pd.DataFrame({"statistic": list(rows.keys()), name: list(rows.values())})


def bucket_counts(s: pd.Series, bins: list, labels: list, name: str) -> pd.DataFrame:
    """Bucket a size distribution and report count + share, cumulative share."""
    cat = pd.cut(s, bins=[b - 0.5 for b in bins[:-1]] + [bins[-1]], labels=labels)
    counts = cat.value_counts().reindex(labels).fillna(0).astype(int)
    total = counts.sum()
    df = pd.DataFrame(
        {
            name: labels,
            "count": counts.values,
            "share": counts.values / max(total, 1),
        }
    )
    df["cumulative_share"] = df["share"].cumsum()
    return df


def bar_figure(x, y, xlabel: str, ylabel: str, title: str, hline: float | None = None):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([str(v) for v in x], y, color="#3b6ea5")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if hline is not None:
        ax.axhline(hline, ls="--", c="crimson", lw=1, label=f"reference = {hline:g}")
        ax.legend()
    ax.grid(axis="y", alpha=0.3)
    return fig
