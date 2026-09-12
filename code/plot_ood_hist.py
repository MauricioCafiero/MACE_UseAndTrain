"""Histograms of the latent OOD signal: stock foundation model vs finetuned.

Draws the distribution of per-frame mean OOD for the rotaxane trajectory and
the held-out dethreading frames, stock vs finetuned, with the 0.25 trust line.
Reads the scores written by ``modal_ood.py`` / ``replay_ood.py``.

    python plot_ood_hist.py            # -> code/viz/replay/ood_hist.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_REPO = Path(__file__).resolve().parent.parent
SCORES = {"off-medium": _REPO / "data" / "replay_ood_scores_off-medium.json",
          "off-large": _REPO / "data" / "replay_ood_scores.json"}
OUT = _REPO / "code" / "viz" / "replay" / "ood_hist.png"

STOCK_C, FT_C = "#8c8c8c", "#1f6fb4"


def main(scores=SCORES, out=OUT):
    sizes = [(k, json.loads(v.read_text())) for k, v in scores.items() if v.exists()]
    panels = [("rot250", "Rotaxane trajectory (250 frames)"),
              ("dethread10", "Held-out dethreading (10 frames)")]

    fig, axes = plt.subplots(len(sizes), 2, figsize=(11, 4.1 * len(sizes)),
                             squeeze=False)
    for row, (size, d) in enumerate(sizes):
      for ax, (key, title) in zip(axes[row], panels):
        b = np.array(d[key]["before"])
        a = np.array(d[key]["after"])
        lo, hi = min(b.min(), a.min()), max(b.max(), a.max())
        pad = 0.05 * (hi - lo) if hi > lo else 0.01
        bins = np.linspace(lo - pad, hi + pad, 24 if len(b) > 50 else 10)
        ax.hist(b, bins=bins, color=STOCK_C, alpha=0.75,
                label=f"stock {size}  (mean {b.mean():.3f})")
        ax.hist(a, bins=bins, color=FT_C, alpha=0.75,
                label=f"finetuned  (mean {a.mean():.3f})")
        ax.axvline(b.mean(), color=STOCK_C, ls="--", lw=1.4)
        ax.axvline(a.mean(), color=FT_C, ls="--", lw=1.4)
        ax.set_title(f"{size}: {title}", fontsize=10.5)
        ax.set_xlabel("mean per-atom latent OOD")
        ax.set_ylabel("frames")
        ax.legend(fontsize=8.5, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Latent OOD: stock foundation model vs GFN2-finetuned "
                 "(matched 78,046-atom pools, TRUST below 0.25)", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    print("wrote", out)
    for size, d in sizes:
        for key in ("rot250", "dethread10", "benchmarks"):
            b, a = np.array(d[key]["before"]), np.array(d[key]["after"])
            print(f"  {size:10s} {key:12s} stock {b.mean():.3f}  ft {a.mean():.3f}  "
                  f"delta {a.mean() - b.mean():+.3f}")


if __name__ == "__main__":
    main()
