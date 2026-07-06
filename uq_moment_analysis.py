#!/usr/bin/env python3
"""Compare Taylor-surrogate (TSE) moment propagation against a Monte Carlo
reference for the maximum temperature of the OTI heat solve.

Inputs (written by uq_max_temperature):
    <dir>/qoi_jet.csv     truncated-Taylor jet of Q = max temperature at the
                          nominal (alpha, A, sigma); coeff = c[alpha] (normalized).
    <dir>/mc_samples.csv  one true-resolve max temperature per Monte Carlo draw.
    <dir>/uq_config.csv   run metadata.

The order-1 surrogate is the |alpha|<=1 subset of the same jet; the order-2
surrogate uses all terms. Gaussian input moments are propagated through each
surrogate EXACTLY via tensor Gauss-Hermite quadrature (the degree-2 polynomial
and its powers are integrated exactly) -- i.e. the central moments are
substituted into the TSE, with no sampling.
"""
import sys
import pathlib
import numpy as np
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

OUT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "uq_output")


def read_config(path):
    cfg = {}
    for line in path.read_text().splitlines()[1:]:
        k, v = line.split(",", 1)
        try:
            cfg[k] = float(v)
        except ValueError:
            cfg[k] = v
    return cfg


def read_jet(path):
    terms = []  # (a0, a1, a2, coeff)
    for line in path.read_text().splitlines()[1:]:
        a0, a1, a2, c = line.split(",")
        terms.append((int(a0), int(a1), int(a2), float(c)))
    return terms


def surrogate_eval(terms, X, max_order):
    """Evaluate sum_alpha c[alpha] * prod X_i^alpha_i, keeping |alpha| <= max_order.
    X has shape (..., 3)."""
    out = np.zeros(X.shape[:-1])
    for a0, a1, a2, c in terms:
        if a0 + a1 + a2 > max_order:
            continue
        out = out + c * X[..., 0] ** a0 * X[..., 1] ** a1 * X[..., 2] ** a2
    return out


def gh_moments(terms, s, max_order, n=10):
    """Central moments (mean, var, skew, excess-kurtosis) of the surrogate under
    independent N(0, s_i^2) inputs, by tensor Gauss-Hermite quadrature."""
    z, w = np.polynomial.hermite_e.hermegauss(n)  # weight exp(-z^2/2)
    w = w / np.sqrt(2.0 * np.pi)                   # normalize to N(0,1) expectation
    Z0, Z1, Z2 = np.meshgrid(z, z, z, indexing="ij")
    W = np.einsum("i,j,k->ijk", w, w, w)
    X = np.stack([Z0 * s[0], Z1 * s[1], Z2 * s[2]], axis=-1)
    Q = surrogate_eval(terms, X, max_order)
    m1 = np.sum(W * Q)
    var = np.sum(W * (Q - m1) ** 2)
    m3 = np.sum(W * (Q - m1) ** 3)
    m4 = np.sum(W * (Q - m1) ** 4)
    std = np.sqrt(var)
    return dict(mean=m1, std=std, var=var,
                skew=m3 / var ** 1.5, exkurt=m4 / var ** 2 - 3.0)


def sample_moments(x):
    n = len(x)
    m = x.mean()
    d = x - m
    var = d.var(ddof=1)
    std = np.sqrt(var)
    m2 = (d ** 2).mean()
    skew = (d ** 3).mean() / m2 ** 1.5
    exkurt = (d ** 4).mean() / m2 ** 2 - 3.0
    return dict(mean=m, std=std, var=var, skew=skew, exkurt=exkurt,
                se_mean=std / np.sqrt(n), se_skew=np.sqrt(6.0 / n),
                se_exkurt=np.sqrt(24.0 / n), n=n)


def main():
    cfg = read_config(OUT / "uq_config.csv")
    terms = read_jet(OUT / "qoi_jet.csv")
    mc = np.loadtxt(OUT / "mc_samples.csv", skiprows=1)

    mu = np.array([cfg["alpha0"], cfg["amplitude0"], cfg["sigma0"]])
    s = cfg["cov"] * mu
    Q0 = cfg["nominal_peak_value"]

    o1 = gh_moments(terms, s, max_order=1)
    o2 = gh_moments(terms, s, max_order=2)
    ref = sample_moments(mc)

    # ---- table -----------------------------------------------------------
    def relerr(a, b):
        return abs(a - b) / abs(b) * 100.0 if b != 0 else float("nan")

    print(f"\nQoI = max temperature. Nominal Q0 = {Q0:.6g}")
    print(f"Inputs: alpha~N({mu[0]:g},({s[0]:.4g})^2)  A~N({mu[1]:g},({s[1]:.4g})^2)  "
          f"sigma~N({mu[2]:g},({s[2]:.4g})^2), independent, CoV={cfg['cov']:.0%}")
    print(f"Monte Carlo reference: N={ref['n']:.0f} true re-solves\n")
    hdr = f"{'moment':<14}{'MC (truth)':>16}{'order-1 TSE':>16}{'order-2 TSE':>16}" \
          f"{'o1 err%':>10}{'o2 err%':>10}"
    print(hdr)
    print("-" * len(hdr))
    rows = [
        ("mean", "mean"),
        ("std dev", "std"),
        ("skewness", "skew"),
        ("excess kurt", "exkurt"),
    ]
    for label, key in rows:
        r, a, b = ref[key], o1[key], o2[key]
        print(f"{label:<14}{r:>16.6g}{a:>16.6g}{b:>16.6g}"
              f"{relerr(a, r):>10.2f}{relerr(b, r):>10.2f}")
    print(f"\nMC standard errors: mean +/-{ref['se_mean']:.3g}, "
          f"skew +/-{ref['se_skew']:.3g}, exkurt +/-{ref['se_exkurt']:.3g}")
    print(f"Order-2 mean shift vs Q0: {o2['mean'] - Q0:+.3g} "
          f"({relerr(o2['mean'], Q0) if Q0 else float('nan'):.2f}% of Q0)")
    out_coV = ref["std"] / ref["mean"] * 100
    print(f"Output CoV (MC): {out_coV:.2f}%  (input CoV {cfg['cov']:.0%})\n")

    # ---- CDFs ------------------------------------------------------------
    # Skewness ~0.14 is nearly invisible in the PDF; the CDF -- and especially
    # the CDF residual against truth -- separates order 1 from order 2 clearly.
    #
    # order-1 surrogate is exactly Gaussian -> analytic CDF.
    # order-2 surrogate CDF comes from the cheap polynomial (visualization only;
    # the reported moments above are from quadrature, not sampling).
    rng = np.random.default_rng(0)
    Xs = rng.standard_normal((2000000, 3)) * s
    q2 = np.sort(surrogate_eval(terms, Xs, 2))

    xs = np.sort(mc)
    n = len(xs)
    grid = np.linspace(mc.min(), mc.max(), 700)
    F_mc = np.searchsorted(xs, grid, side="right") / n          # empirical truth
    F_o1 = norm.cdf(grid, o1["mean"], o1["std"])                # order-1 (Gaussian)
    F_o2 = np.searchsorted(q2, grid, side="right") / len(q2)    # order-2
    band = 2.0 * np.sqrt(np.clip(F_mc * (1 - F_mc), 0, None) / n)  # +/-2 SE of empirical CDF

    ks1 = np.max(np.abs(F_o1 - F_mc))
    ks2 = np.max(np.abs(F_o2 - F_mc))
    print(f"Kolmogorov-Smirnov distance to MC truth: order-1 {ks1:.4f}, "
          f"order-2 {ks2:.4f}  ({ks1 / ks2:.1f}x closer)")

    # KL divergence D(truth || surrogate), from binned bin-probabilities.
    # Bin probabilities come straight from the CDFs: truth from the empirical
    # CDF, order-1 from the exact Gaussian CDF, order-2 from its (finely
    # sampled) CDF. Lower = closer to the target distribution.
    nb = 40
    edges = np.linspace(mc.min(), mc.max(), nb + 1)
    P = np.diff(np.searchsorted(xs, edges, side="right") / n)
    Q1 = np.diff(norm.cdf(edges, o1["mean"], o1["std"]))
    Q2 = np.diff(np.searchsorted(q2, edges, side="right") / len(q2))
    # restrict to truth's support, renormalize, floor model bins to avoid log(0)
    P, Q1, Q2 = P / P.sum(), Q1 / Q1.sum(), Q2 / Q2.sum()
    floor_q = 0.5 / len(q2)
    m = P > 0
    kl1 = float(np.sum(P[m] * np.log(P[m] / np.maximum(Q1[m], floor_q))))
    kl2 = float(np.sum(P[m] * np.log(P[m] / np.maximum(Q2[m], floor_q))))
    kl_floor = (int(m.sum()) - 1) / (2.0 * n)  # finite-sample bias of D from N draws
    print(f"KL divergence D(truth||surrogate) [nats]: order-1 {kl1:.2e}, "
          f"order-2 {kl2:.2e}  ({kl1 / kl2:.1f}x smaller)")
    print(f"  (N={n} finite-sample noise floor ~{kl_floor:.1e} nats)\n")

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))

    # ---- left: CDF overlay with an upper-tail inset ----
    a = ax[0]
    a.plot(grid, F_mc, color="0.45", lw=3.0, alpha=0.7,
           label=f"Monte Carlo truth (N={n})")
    a.plot(grid, F_o1, "C0--", lw=2, label="order-1 TSE (Gaussian)")
    a.plot(grid, F_o2, "C3-", lw=2, label="order-2 TSE")
    a.set_xlabel("maximum temperature  $T_{\\max}$")
    a.set_ylabel("cumulative probability  $F(T_{\\max})$")
    a.set_title("CDF of $T_{\\max}$: TSE surrogate vs. truth")
    a.legend(fontsize=8, loc="upper left")
    a.text(0.03, 0.55,
           "$D_{\\mathrm{KL}}(\\mathrm{truth}\\,\\|\\,\\mathrm{TSE})$\n"
           f"order-1: {kl1:.1e} nats\n"
           f"order-2: {kl2:.1e} nats\n"
           f"({kl1 / kl2:.0f}$\\times$ closer)",
           transform=a.transAxes, fontsize=8, va="top",
           bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9))

    # inset zoom on the upper tail, where the positive skew separates the curves
    axins = inset_axes(a, width="44%", height="44%", loc="lower right",
                       borderpad=1.2)
    axins.plot(grid, F_mc, color="0.45", lw=3, alpha=0.7)
    axins.plot(grid, F_o1, "C0--", lw=2)
    axins.plot(grid, F_o2, "C3-", lw=2)
    axins.set_xlim(np.quantile(mc, 0.85), np.quantile(mc, 0.999))
    axins.set_ylim(0.85, 1.001)
    axins.tick_params(labelsize=6)
    axins.set_title("upper tail (positive skew)", fontsize=7)
    mark_inset(a, axins, loc1=2, loc2=4, fc="none", ec="0.6", lw=0.8)

    # ---- right: relative error of each moment vs. MC truth ----
    # The clearest separation: order-1 is a symmetric Gaussian, so it captures
    # mean/std but reports skewness = excess-kurtosis = 0 (100% error); order-2
    # recovers the shape moments from the same single solve.
    b = ax[1]
    labels = ["mean", "std dev", "skewness", "excess\nkurtosis"]
    keys = ["mean", "std", "skew", "exkurt"]
    e1 = [relerr(o1[k], ref[k]) for k in keys]
    e2 = [relerr(o2[k], ref[k]) for k in keys]
    x = np.arange(len(labels))
    bw = 0.38
    b.bar(x - bw / 2, e1, bw, color="C0", label="order-1 TSE")
    b.bar(x + bw / 2, e2, bw, color="C3", label="order-2 TSE")
    b.set_xticks(x)
    b.set_xticklabels(labels)
    b.set_ylabel("relative error vs. MC truth (%)")
    b.set_ylim(0, 112)
    b.set_title(f"Moment error: one OTI solve vs. {n} re-solves")
    b.legend(fontsize=9, loc="upper left")
    for xi, v in zip(x - bw / 2, e1):
        b.text(xi, v + 1.5, f"{v:.1f}", ha="center", va="bottom", fontsize=8, color="C0")
    for xi, v in zip(x + bw / 2, e2):
        b.text(xi, v + 1.5, f"{v:.1f}", ha="center", va="bottom", fontsize=8, color="C3")

    fig.suptitle("UQ of maximum temperature: Taylor-surrogate (one solve) vs "
                 f"Monte Carlo ({n} re-solves)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    figdir = OUT / "figures"
    figdir.mkdir(exist_ok=True)
    out_png = figdir / "uq_max_temperature.png"
    fig.savefig(out_png, dpi=140)
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
