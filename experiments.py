"""Experiments for model v2: figures + CSV into model2/out/. Run: python experiments.py"""
import csv
import os
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from model import Params, POLICIES, run, replace, draw_scenarios, demand_path, regime_path, simulate

warnings.filterwarnings("ignore", category=RuntimeWarning)
OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)
N = 200
NMAP = 120
B = Params()

SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"        # categorical slots 1-3
YELLOW, VIOLET = "#eda100", "#6250d6"
# naive hardware-first is the neutral reference (grey); the three real choices get slots 1-3
COL = {"hardware_first": INK2, "buy_ahead": BLUE, "gated": ORANGE, "cloud_first": AQUA}
LABEL = {"hardware_first": "Hardware-first", "buy_ahead": "Buy-ahead", "gated": "Gated",
         "cloud_first": "Cloud-first"}
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "axes.axisbelow": True,
})


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=200)
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------------
# 1. One scenario, side by side: demand, data, power, owned GPUs
# ---------------------------------------------------------------------
def fig_path():
    sc = draw_scenarios(400, B.T, seed=7)
    # pick a run with a shortage starting during the ramp
    for i in range(400):
        s = regime_path(B, sc["u"][i])
        if s[6:16].any() and s.sum() < 20:
            break
    M = demand_path(B, sc["zD"][i], sc["zt0"][i], sc["zm"][i])
    t = np.arange(B.T)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, pol in zip(axes, ("hardware_first", "gated")):
        h = simulate(pol, B, M, s)["hist"]
        for a, b in _runs(s):
            ax.axvspan(a - 0.5, b + 0.5, color=GRID, alpha=0.6, lw=0)
        ax.plot(t, M, color=INK2, lw=1.5, ls="--", label="Demand signal")
        ax.plot(t, h["A"], color=AQUA, lw=2, label="Data readiness")
        ax.step(t, h["E"], where="post", color=VIOLET, lw=2, label="Power")
        ax.step(t, h["K"], where="post", color=COL[pol], lw=2, label="Owned GPUs")
        ax.fill_between(t, h["U"], h["K"], step="post", color=COL[pol], alpha=0.15, lw=0)
        ax.set(title=LABEL[pol], xlabel="Month")
    axes[0].set_ylabel("GPU-eq")
    axes[0].legend(frameon=False, loc="upper left", fontsize=8)
    axes[1].text(0.99, 0.02, "grey bands = GPU shortage; shaded = owned but not used",
                 transform=axes[1].transAxes, ha="right", fontsize=8, color=INK2)
    save(fig, "fig1_example_path.png")


def _runs(s):
    out, start = [], None
    for k, x in enumerate(s):
        if x and start is None:
            start = k
        if not x and start is not None:
            out.append((start, k - 1)); start = None
    if start is not None:
        out.append((start, len(s) - 1))
    return out


# ---------------------------------------------------------------------
# 2. Baseline table + decomposition hardware_first -> gated
# ---------------------------------------------------------------------
STEPS = [
    ("Hardware-first", B, "hardware_first"),
    ("- no shortage gaming", replace(B, gaming=False), "hardware_first"),
    ("+ power built ahead", replace(B, gaming=False, hf_power_ahead=True), "hardware_first"),
    ("+ data built ahead = Buy-ahead", B, "buy_ahead"),
    ("+ GPUs capped by readiness = Gated", B, "gated"),
    ("Cloud-first", B, "cloud_first"),
]
COLS = ("profit", "value", "total_cost", "c_gpu", "c_power", "c_data", "c_energy", "c_opex",
        "c_cloud", "salvage_pv", "served_share", "readiness_gap_share", "lost_share", "util",
        "idle_share", "dark_share", "gpus_bought", "peak_excess", "bullwhip", "co2_t",
        "co2_op_t", "co2_emb_t", "co2_waste_t", "co2_per_useful_gpuh")


def baseline():
    rows, res = [], []
    for name, p, pol in STEPS:
        v = run(p, n=N, seed=0, policies=(pol,))[pol]
        res.append((name, pol, v))
        row = {"variant": name}
        for k in COLS:
            row[k] = round(float(np.nanmean(v[k])), 3)
        row["profit_p10"] = round(float(np.percentile(v["profit"], 10)))
        row["profit_p90"] = round(float(np.percentile(v["profit"], 90)))
        rows.append(row)
    with open(os.path.join(OUT, "baseline_table.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print("wrote baseline_table.csv")

    fig, ax = plt.subplots(figsize=(8, 4))
    y = np.arange(len(res))[::-1]
    for yi, (name, pol, v) in zip(y, res):
        m = v["profit"].mean() / 1e6
        ax.barh(yi, m, height=0.6, color=COL[pol])
        ax.text(m + 0.15, yi, f"{m:.1f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(y, [r[0] for r in res])
    ax.set(xlabel="Mean NPV profit over 48 months (USD million)",
           title="Hardware-first to gated, one change at a time")
    ax.grid(axis="y", visible=False)
    save(fig, "fig2_decomposition.png")
    return res


# ---------------------------------------------------------------------
# 3. What owned GPUs do: used / idle (no data) / dark (no power)
# ---------------------------------------------------------------------
def fig_gap(res):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    owned = [r for r in res if r[1] != "cloud_first"]
    y = np.arange(len(owned))[::-1]
    parts = (("util", "Used", BLUE), ("idle_share", "Powered, no workload", YELLOW),
             ("dark_share", "No power (dark)", INK2))
    for yi, (name, pol, v) in zip(y, owned):
        left = 0
        for k, lab, c in parts:
            w = np.nanmean(v[k])
            ax.barh(yi, w, left=left, height=0.6, color=c, edgecolor=SURF, linewidth=2)
            if w > 0.06:
                ax.text(left + w / 2, yi, f"{w:.0%}", ha="center", va="center", fontsize=8,
                        color=INK if c == YELLOW else SURF)
            left += w
    ax.set_yticks(y, [r[0] for r in owned])
    ax.set(xlim=(0, 1), xlabel="Share of owned GPU-months",
           title="The deployment gap inside the fleet")
    ax.legend(handles=[Patch(color=c, label=l) for _, l, c in parts], frameon=False,
              ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=8)
    ax.grid(axis="y", visible=False)
    save(fig, "fig3_fleet_use.png")


# ---------------------------------------------------------------------
# 4. Bullwhip: orders vs demand growth, averaged over runs
# ---------------------------------------------------------------------
def fig_bullwhip():
    p = B
    sc = draw_scenarios(N, p.T, seed=0)
    t = np.arange(p.T)
    acc = {pol: np.zeros(p.T) for pol in ("hardware_first", "buy_ahead", "gated")}
    dM = np.zeros(p.T)
    for i in range(N):
        M = demand_path(p, sc["zD"][i], sc["zt0"][i], sc["zm"][i])
        s = regime_path(p, sc["u"][i])
        dM[1:] += np.diff(M) / N
        for pol in acc:
            acc[pol] += simulate(pol, p, M, s)["orders"] / N
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [2, 1]})
    ax = axes[0]
    ax.plot(t, dM, color=INK2, lw=1.5, ls="--", label="Demand growth per month")
    for pol, a in acc.items():
        ax.plot(t, a, color=COL[pol], lw=2, label=f"{LABEL[pol]} GPU orders")
    ax.set(xlabel="Month", ylabel="GPU-eq per month", title="Orders vs demand growth (mean of runs)")
    ax.legend(frameon=False, fontsize=8)

    # glut vs shortage frequency
    ax = axes[1]
    xs = np.array([0.0, 0.02, 0.04, 0.08, 0.12, 0.16])
    for name, kw, pol, ls in (("Hardware-first", {}, "hardware_first", "-"),
                              ("Hardware-first, no gaming", {"gaming": False}, "hardware_first", ":"),
                              ("Buy-ahead", {}, "buy_ahead", "-"),
                              ("Gated", {}, "gated", "-")):
        ys = [run(replace(p, p_ns=x, **kw), n=NMAP, seed=0, policies=(pol,))[pol]["peak_excess"].mean()
              for x in xs]
        ax.plot(xs, ys, color=COL[pol], lw=2, ls=ls, marker="o", ms=4, label=name)
    ax.set(xlabel="Monthly probability a shortage starts", ylabel="Peak owned GPUs above usable workload",
           title="Glut grows with shortages")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "fig4_bullwhip.png")


# ---------------------------------------------------------------------
# 5-6. Winner maps
# ---------------------------------------------------------------------
def winner_map(xname, xs, yname, ys, setter, fname, title, xlabel, ylabel):
    W = np.zeros((len(ys), len(xs)), int)
    for i, yv in enumerate(ys):
        for j, xv in enumerate(xs):
            out = run(setter(xv, yv), n=NMAP, seed=0)
            m = [out[pol]["profit"].mean() for pol in POLICIES]
            W[i, j] = int(np.argmax(m))
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    cmap = ListedColormap([COL[p] for p in POLICIES])
    ax.imshow(W, origin="lower", cmap=cmap, vmin=0, vmax=len(POLICIES) - 1, aspect="auto")
    for i in range(len(ys) + 1):
        ax.axhline(i - 0.5, color=SURF, lw=2)
    for j in range(len(xs) + 1):
        ax.axvline(j - 0.5, color=SURF, lw=2)
    ax.set_xticks(range(len(xs)), [f"{x:g}" for x in xs])
    ax.set_yticks(range(len(ys)), [f"{y:g}" for y in ys])
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    ax.grid(False)
    ax.legend(handles=[Patch(color=COL[p], label=LABEL[p]) for p in POLICIES], frameon=False,
              loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4, fontsize=8)
    save(fig, fname)
    return W


def maps():
    vs = np.array([3, 6, 12, 18, 24, 36, 48])
    pns = np.array([0.0, 0.02, 0.04, 0.08, 0.12, 0.16])
    winner_map("v", vs, "p_ns", pns,
               lambda v, x: replace(B, v=float(v), p_ns=float(x)),
               "fig5_winner_value_vs_shortage.png",
               "Most profitable policy: value of compute vs shortage risk",
               "Value per useful GPU-hour (USD)", "Monthly probability a GPU shortage starts")
    caps = np.array([0.0, 0.1, 0.25, 0.5, 1.0])
    winner_map("cap", caps, "p_ns", pns,
               lambda c, x: replace(B, v=18.0, cloud_cap_short=float(c), p_ns=float(x)),
               "fig6_winner_cloudcap_vs_shortage.png",
               "Best policy at v = $18/h: cloud in shortage vs shortage risk",
               "Cloud available during a shortage (share of planned peak)",
               "Monthly probability a GPU shortage starts")


def fig_advantage():
    """Gated minus hardware-first profit, vs data readiness speed, for three values of v."""
    months = np.array([4, 8, 12, 16, 24, 32])
    fig, ax = plt.subplots(figsize=(7, 4))
    for v, ls in ((4.0, ":"), (8.0, "--"), (16.0, "-")):
        d = []
        for mo in months:
            out = run(replace(B, v=v, a_rate=B.D_max / mo), n=NMAP, seed=0,
                      policies=("hardware_first", "gated"))
            d.append((out["gated"]["profit"] - out["hardware_first"]["profit"]).mean() / 1e6)
        ax.plot(months, d, color=ORANGE, lw=2, ls=ls, marker="o", ms=4)
        ax.text(months[-1] + 0.5, d[-1], f"v = ${v:g}/h", va="center", fontsize=8, color=INK2)
    ax.axhline(0, color=INK2, lw=1)
    ax.set(xlabel="Months to build data readiness for peak demand", xlim=(2, 38),
           ylabel="Gated minus hardware-first profit (USD m)",
           title="Gating pays most when readiness is slow")
    save(fig, "fig7_gated_advantage.png")


# ---------------------------------------------------------------------
# 8. Carbon per useful GPU-hour: operational vs embodied, and wasted carbon
# ---------------------------------------------------------------------
def fig_carbon(res):
    fig, ax = plt.subplots(figsize=(8, 3.8))
    y = np.arange(len(res))[::-1]
    for yi, (name, pol, v) in zip(y, res):
        per = v["co2_per_useful_gpuh"]                          # kg per useful GPU-hour, per run
        op = np.nanmean(per * v["co2_op_t"] / v["co2_t"])
        emb = np.nanmean(per * v["co2_emb_t"] / v["co2_t"])
        ax.barh(yi, op, height=0.6, color=BLUE, edgecolor=SURF, linewidth=2)
        ax.barh(yi, emb, left=op, height=0.6, color=YELLOW, edgecolor=SURF, linewidth=2)
        waste = v["co2_waste_t"].mean()
        ax.text(op + emb + 0.01, yi, f"{op + emb:.2f} kg  (wasted: {waste:,.0f} t)", va="center",
                fontsize=8, color=INK2)
    ax.set_yticks(y, [r[0] for r in res])
    ax.set(xlabel="kgCO2e per useful GPU-hour", xlim=(0, None),
           title="Carbon per useful GPU-hour: operational + embodied")
    ax.legend(handles=[Patch(color=BLUE, label="Operational (electricity)"),
                       Patch(color=YELLOW, label="Embodied (manufacturing, amortised)")],
              frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=8)
    ax.grid(axis="y", visible=False)
    ax.margins(x=0.35)
    save(fig, "fig8_carbon.png")


if __name__ == "__main__":
    fig_path()
    res = baseline()
    fig_gap(res)
    fig_carbon(res)
    fig_bullwhip()
    maps()
    fig_advantage()
