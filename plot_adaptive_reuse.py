#!/usr/bin/env python3
"""Plot adaptive surrogate reuse: the validity gate decides reuse-vs-re-solve as
a parameter drifts, covering a whole sweep with a few certified solves.

Usage: plot_adaptive_reuse.py <reuse_dir> <out.png>
"""
import sys
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d, out = sys.argv[1], sys.argv[2]
summ = {k: float(v) for k, v in (r for r in csv.reader(open(f"{d}/reuse_summary.csv")) if r[0] != "key")}
rows = list(csv.DictReader(open(f"{d}/reuse_sweep.csv")))
al = np.array([float(r["alpha"]) for r in rows])
truth = np.array([float(r["truth"]) for r in rows])
pred = np.array([float(r["pred"]) for r in rows])
err = np.array([float(r["abs_err"]) for r in rows])
budget = np.array([float(r["budget"]) for r in rows])
is_anchor = np.array([int(r["is_anchor"]) for r in rows])
anchor_alpha = np.array([float(r["anchor_alpha"]) for r in rows])

tau = summ["tau"]
solves = int(summ["solves"])
nq = int(summ["queries"])
fp = int(summ["false_positives"])

anchors = sorted(set(anchor_alpha))
colors = plt.cm.viridis(np.linspace(0.1, 0.85, len(anchors)))
cmap = {a: colors[i] for i, a in enumerate(anchors)}

fig, ax = plt.subplots(2, 1, figsize=(10.5, 7.4), sharex=True,
                       gridspec_kw=dict(height_ratios=[2, 1]))

# ---- top: QoI vs alpha, true curve + reused/anchor predictions ----
a = ax[0]
# shade each anchor's reuse interval
for an in anchors:
    seg = al[anchor_alpha == an]
    a.axvspan(seg.min(), seg.max(), color=cmap[an], alpha=0.10, zorder=0)
a.plot(al, truth, "k", lw=2.2, zorder=2, label="true QoI (dense PDE sweep)")
reuse_mask = is_anchor == 0
a.scatter(al[reuse_mask], pred[reuse_mask], s=14, c=[cmap[a_] for a_ in anchor_alpha[reuse_mask]],
          zorder=3, label=f"reused from surrogate ({int(reuse_mask.sum())} pts, 0 solves)")
a.scatter(al[is_anchor == 1], pred[is_anchor == 1], marker="*", s=320, c="C3",
          edgecolor="k", zorder=5, label=f"anchor: PDE re-solve ({solves} solves)")
a.set_ylabel("QoI: sensor temperature", fontsize=12)
a.set_title(f"Adaptive surrogate reuse: {nq} queries covered by {solves} PDE solves "
            f"({nq / solves:.0f}× fewer)", fontsize=12.5)
a.legend(fontsize=9, loc="upper right")

# ---- bottom: reuse error vs the tau budget ----
b = ax[1]
b.plot(al, budget, color="C3", ls=":", lw=2, label=rf"trust budget $\tau|f|$ ($\tau={tau:g}$)")
b.scatter(al[reuse_mask], err[reuse_mask], s=14,
          c=[cmap[a_] for a_ in anchor_alpha[reuse_mask]], zorder=3, label="reuse error |pred − truth|")
b.scatter(al[is_anchor == 1], err[is_anchor == 1], marker="*", s=200, c="C3", edgecolor="k",
          zorder=5, label="anchor (exact)")
b.set_xlabel(r"$\alpha$  (re-estimated diffusivity, over operational time)", fontsize=12)
b.set_ylabel("abs. error", fontsize=12)
b.set_title(f"Every reuse stays under budget — false positives: {fp}/{int(reuse_mask.sum())}", fontsize=12)
b.legend(fontsize=9, loc="upper left")

fig.suptitle("On-the-fly error analysis at work: the validity gate certifies when the surrogate "
             "can be reused", fontsize=13)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(out, dpi=140)
print("Wrote", out)
