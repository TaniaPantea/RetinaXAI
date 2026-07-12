
from collections import OrderedDict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt



METRIC_SPECS = {
    "fidelity_top5":  dict(label="Fidelity top-5%",  lo=0.0, hi=1.0, higher_better=True),
    "fidelity_top10": dict(label="Fidelity top-10%", lo=0.0, hi=1.0, higher_better=True),
    "fidelity_top15": dict(label="Fidelity top-15%", lo=0.0, hi=1.0, higher_better=True),
    "fidelity_top20": dict(label="Fidelity top-20%", lo=0.0, hi=1.0, higher_better=True),
    "insertion_auc":  dict(label="Insertion AUC",    lo=0.0, hi=1.0, higher_better=True),
    "deletion_auc":   dict(label="Deletion AUC",     lo=0.0, hi=1.0, higher_better=False),
    "stability":      dict(label="Stability (SSIM)", lo=0.0, hi=1.0, higher_better=True),
    "stability_corr": dict(label="Stability (Pearson)", lo=-1.0, hi=1.0, higher_better=True,
                           note="negative = explanation flips under small "
                                "perturbation (unstable)"),
    "dice": dict(label="Dice", lo=0.0, hi=1.0, higher_better=True),
    "iou":  dict(label="IoU",  lo=0.0, hi=1.0, higher_better=True),
}

CATEGORIES = OrderedDict([
    ("Fidelity", ["fidelity_top5", "fidelity_top10", "fidelity_top15",
                  "fidelity_top20", "insertion_auc", "deletion_auc"]),
    ("Stability", ["stability", "stability_corr"]),
    ("Localisation", ["dice", "iou"]),
])

CATEGORY_LEGENDS = {
    "Fidelity": (
        "Shows how much the prediction depends on the pixels highlighted by the CAM. "
        "Higher Fidelity and Insertion AUC values are better. Lower Deletion AUC values are better."
    ),
    "Stability": (
        "Shows how consistent the explanation remains under small input perturbations. "
        "Higher SSIM and Pearson values are better. A negative Pearson value indicates an unstable explanation."
    ),
    "Localisation": (
        "Shows how well the CAM overlaps the region marked by the doctor. "
        "Higher Dice and IoU values are better. IoU is the stricter metric."
    ),
}


def category_of(key: str):
    for cat, keys in CATEGORIES.items():
        if key in keys:
            return cat
    return None


def present_metrics_by_category(available: dict):
    out = OrderedDict()
    for cat, keys in CATEGORIES.items():
        rows = []
        for k in keys:
            v = available.get(k)
            if v is None:
                continue
            try:
                rows.append((k, float(v)))
            except (TypeError, ValueError):
                continue
        if rows:
            out[cat] = rows
    return out



def _fig_height(n_rows: int) -> float:
    return max(1.2, 0.55 * n_rows + 0.6)


def category_bar_figure(rows, title=None):
    n = len(rows)
    fig, ax = plt.subplots(figsize=(6.4, _fig_height(n)))

    y_positions = list(range(n))[::-1]

    for y, (key, value) in zip(y_positions, rows):
        spec = METRIC_SPECS.get(key, dict(label=key, lo=0.0, hi=1.0,
                                          higher_better=True))
        lo, hi = spec["lo"], spec["hi"]
        span = hi - lo if hi > lo else 1.0

        ax.barh(y, span, left=lo, height=0.5, color="#e6e6e6",
                edgecolor="#cccccc", zorder=1)

        v = float(np.clip(value, lo, hi))
        if spec["higher_better"]:
            ax.barh(y, v - lo, left=lo, height=0.5, color="#4c78a8", zorder=2)
        else:
            ax.barh(y, hi - v, left=v, height=0.5, color="#4c78a8", zorder=2)

        ax.plot([v, v], [y - 0.28, y + 0.28], color="#22303f", lw=2, zorder=3)
        ax.text(hi + 0.02 * span, y, f"{value:.3f}", va="center", ha="left",
                fontsize=9, color="#22303f")

        arrow = "→ better" if spec["higher_better"] else "better ←"
        ax.text(lo - 0.02 * span, y, arrow, va="center", ha="right",
                fontsize=7.5, color="#888888")

    ax.set_yticks(y_positions)
    ax.set_yticklabels([METRIC_SPECS.get(k, {}).get("label", k) for k, _ in rows],
                       fontsize=9)
    pad = 0.30
    xmins = [METRIC_SPECS.get(k, {}).get("lo", 0.0) for k, _ in rows]
    xmaxs = [METRIC_SPECS.get(k, {}).get("hi", 1.0) for k, _ in rows]
    ax.set_xlim(min(xmins) - pad, max(xmaxs) + pad)
    ax.set_ylim(-0.6, (n - 1) + 0.6)
    ax.set_xlabel("metric value on its theoretical scale", fontsize=8,
                  color="#666666")
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="both", length=0)
    if title:
        ax.set_title(title, fontsize=11, loc="left", color="#22303f")
    fig.tight_layout()
    return fig