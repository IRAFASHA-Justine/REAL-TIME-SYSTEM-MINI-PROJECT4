"""rt_edge_cloud_sim.py  --  Mini Project 4.3
Edge-versus-cloud latency simulator for a repeated real-time IoT decision task
(Section 4.9 edge-computing rationale, Section 4.8 URLLC bounded latency).

A sensor stream at a fixed rate produces decision events. Each event is
processed by (a) an edge path: negligible transport, local compute on limited
hardware, (b) a cloud path: upload leg + remote compute + download leg, where
the network legs have a heavy (Pareto) congestion tail, and (c) a hybrid policy.
Pure Python + matplotlib.   Run:  python rt_edge_cloud_sim.py
Outputs: figures (PNG), results_table.csv, sweep_results.csv in ./output_4_3/
"""
import csv
import math
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------- parameters
N_EVENTS = 20000          # decision events per run
RATE_HZ = 20.0            # nominal sensor rate (one event every 50 ms)
DEADLINE = 0.100          # end-to-end deadline per decision (s)
P_HARD = 0.20             # fraction of "hard" decisions (e.g. anomaly suspected)
BASE_CONGESTION = 0.2     # baseline congestion level for the results table
SEED = 7
OUT_DIR = "output_4_3"

# edge: local hardware -> low variance, but slower on hard decisions (seconds)
EDGE_TRANSPORT = (0.0005, 0.0015)          # negligible on-device transport
EDGE_EASY = (0.020, 0.003)                 # (mean, std) compute
EDGE_HARD = (0.090, 0.012)

# cloud: powerful server -> fast compute, but a variable, heavy-tailed network
CLOUD_COMPUTE_EASY = (0.008, 0.002)
CLOUD_COMPUTE_HARD = (0.015, 0.003)
LEG_MEDIAN = 0.012        # median one-way network delay at zero congestion (s)
LEG_SIGMA = 0.35          # lognormal spread of the one-way delay
CONGESTION_SLOWDOWN = 1.5 # median grows by (1 + 1.5 * congestion)
SPIKE_P0, SPIKE_PC = 0.005, 0.12   # spike prob per leg = P0 + PC * congestion
SPIKE_SCALE, SPIKE_ALPHA = 0.035, 1.8   # Pareto spike: min 35 ms, heavy tail


# ---------------------------------------------------------------- latency models
def edge_latency(hard, rng):
    """One end-to-end edge latency sample (s): transport + local compute."""
    mean, std = EDGE_HARD if hard else EDGE_EASY
    compute = max(0.002, mean + std * rng.gauss(0, 1))
    return rng.uniform(*EDGE_TRANSPORT) + compute


def _leg(congestion, rng):
    """One network leg: lognormal base delay plus occasional Pareto spike.
    All variates are always drawn so the random sequence is identical for every
    congestion level (common random numbers -> smooth, comparable sweeps)."""
    z, u, v = rng.gauss(0, 1), rng.random(), rng.random()
    base = LEG_MEDIAN * (1 + CONGESTION_SLOWDOWN * congestion) * math.exp(LEG_SIGMA * z)
    spike = SPIKE_SCALE * (1 - v) ** (-1 / SPIKE_ALPHA)
    p_spike = SPIKE_P0 + SPIKE_PC * congestion
    return base + (spike if u < p_spike else 0.0)


def cloud_latency(congestion, hard, rng):
    """One end-to-end cloud latency sample (s): upload + remote compute + download."""
    up = _leg(congestion, rng)
    mean, std = CLOUD_COMPUTE_HARD if hard else CLOUD_COMPUTE_EASY
    compute = max(0.002, mean + std * rng.gauss(0, 1))
    down = _leg(congestion, rng)
    return up + compute + down


# ------------------------------------------------------------------- simulation
def make_events(n=N_EVENTS, seed=SEED):
    """Shared event stream: (timestamp, is_hard). Both paths see the same events."""
    rng = random.Random(seed)
    return [(k / RATE_HZ, rng.random() < P_HARD) for k in range(n)]


def simulate(events, congestion, seed=SEED):
    edge_rng = random.Random(seed + 1)
    cloud_rng = random.Random(seed + 2)
    edge = [edge_latency(h, edge_rng) for _, h in events]
    cloud = [cloud_latency(congestion, h, cloud_rng) for _, h in events]
    # hybrid policy: easy decisions stay on the edge, hard ones go to the cloud
    hybrid = [c if h else e for (_, h), e, c in zip(events, edge, cloud)]
    return edge, cloud, hybrid


# ------------------------------------------------------------------- statistics
def percentile(sorted_vals, q):
    k = (len(sorted_vals) - 1) * q / 100
    f, c = math.floor(k), math.ceil(k)
    return sorted_vals[f] + (sorted_vals[c] - sorted_vals[f]) * (k - f)


def stats(lat):
    s = sorted(lat)
    return dict(mean=sum(s) / len(s), p95=percentile(s, 95), p99=percentile(s, 99),
                worst=s[-1], comp=100 * sum(x <= DEADLINE for x in s) / len(s))


# ------------------------------------------------------------------------ plots
def plot_distributions(edge, cloud, fname):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    cap = 0.300
    bins = [i * 0.005 for i in range(int(cap / 0.005) + 1)]
    ax1.hist([min(x, cap) for x in edge], bins=bins, alpha=0.6, color="tab:blue", label="edge")
    ax1.hist([min(x, cap) for x in cloud], bins=bins, alpha=0.6, color="tab:red",
             label=f"cloud (congestion {BASE_CONGESTION})")
    ax1.axvline(DEADLINE, color="k", ls="--", label="deadline (100 ms)")
    ax1.set_yscale("log")
    ax1.set_xlabel("end-to-end latency (s), values above 0.3 s collected in last bin")
    ax1.set_ylabel("events (log scale)")
    ax1.set_title("Figure 1a - Latency histograms")
    ax1.legend()
    for lat, col, name in ((edge, "tab:blue", "edge"), (cloud, "tab:red", "cloud")):
        s = sorted(lat)
        ax2.plot(s, [(i + 1) / len(s) for i in range(len(s))], color=col, label=name)
    ax2.axvline(DEADLINE, color="k", ls="--", label="deadline")
    ax2.set_xscale("log")
    ax2.set_xlabel("end-to-end latency (s, log scale)")
    ax2.set_ylabel("empirical CDF")
    ax2.set_title("Figure 1b - Empirical CDFs (the cloud tail is visible on the right)")
    ax2.legend(loc="lower right")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=130)
    plt.close(fig)


def plot_sweep(levels, cloud_c, hybrid_c, edge_c, fname):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(levels, cloud_c, "o-", color="tab:red", label="cloud")
    ax.plot(levels, hybrid_c, "s-", color="tab:green", label="hybrid")
    ax.axhline(edge_c, color="tab:blue", ls="--", label=f"edge (flat, {edge_c:.1f}%)")
    ax.set_xlabel("cloud congestion level")
    ax.set_ylabel("deadline compliance (%)")
    ax.set_title("Figure 2 - Deadline compliance versus cloud congestion")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=130)
    plt.close(fig)


def plot_hybrid(edge, cloud, hybrid, fname):
    fig, ax = plt.subplots(figsize=(9, 5))
    for lat, col, name in ((edge, "tab:blue", "pure edge"), (cloud, "tab:red", "pure cloud"),
                           (hybrid, "tab:green", "hybrid")):
        s = sorted(lat)
        ax.plot(s, [(i + 1) / len(s) for i in range(len(s))], color=col, label=name)
    ax.axvline(DEADLINE, color="k", ls="--", label="deadline")
    ax.set_xscale("log")
    ax.set_xlabel("end-to-end latency (s, log scale)")
    ax.set_ylabel("empirical CDF")
    ax.set_title(f"Figure 3 - Edge, cloud and hybrid latency (congestion {BASE_CONGESTION})")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------------- main
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    events = make_events()

    # ---- results at the baseline congestion level
    edge, cloud, hybrid = simulate(events, BASE_CONGESTION)
    hdr = ["architecture", "mean(ms)", "p95(ms)", "p99(ms)", "worst(ms)", "compliance(%)"]
    rows = []
    for name, lat in (("edge", edge), (f"cloud (c={BASE_CONGESTION})", cloud),
                      (f"hybrid (c={BASE_CONGESTION})", hybrid)):
        s = stats(lat)
        rows.append([name, f"{s['mean'] * 1e3:.1f}", f"{s['p95'] * 1e3:.1f}",
                     f"{s['p99'] * 1e3:.1f}", f"{s['worst'] * 1e3:.1f}", f"{s['comp']:.2f}"])
    print(f"Events: {len(events)}  hard decisions: {sum(h for _, h in events)}  "
          f"deadline: {DEADLINE * 1e3:.0f} ms\n")
    widths = [max(len(r[i]) for r in [hdr] + rows) for i in range(len(hdr))]
    for r in [hdr] + rows:
        print("  ".join(c.ljust(w) for c, w in zip(r, widths)))
    with open(os.path.join(OUT_DIR, "results_table.csv"), "w", newline="") as f:
        csv.writer(f).writerows([hdr] + rows)

    # ---- congestion sweep
    levels = [i / 10 for i in range(11)]
    sweep_hdr = ["congestion", "cloud_comp(%)", "cloud_p99(ms)", "hybrid_comp(%)", "edge_comp(%)"]
    sweep_rows, cloud_c, hybrid_c = [], [], []
    edge_c = stats(edge)["comp"]
    for c in levels:
        _, cl, hy = simulate(events, c)
        sc, sh = stats(cl), stats(hy)
        cloud_c.append(sc["comp"])
        hybrid_c.append(sh["comp"])
        sweep_rows.append([f"{c:.1f}", f"{sc['comp']:.2f}", f"{sc['p99'] * 1e3:.1f}",
                           f"{sh['comp']:.2f}", f"{edge_c:.2f}"])
    print("\nCongestion sweep:")
    widths = [max(len(r[i]) for r in [sweep_hdr] + sweep_rows) for i in range(len(sweep_hdr))]
    for r in [sweep_hdr] + sweep_rows:
        print("  ".join(c.ljust(w) for c, w in zip(r, widths)))
    with open(os.path.join(OUT_DIR, "sweep_results.csv"), "w", newline="") as f:
        csv.writer(f).writerows([sweep_hdr] + sweep_rows)

    plot_distributions(edge, cloud, "fig1_latency_distributions.png")
    plot_sweep(levels, cloud_c, hybrid_c, edge_c, "fig2_compliance_vs_congestion.png")
    plot_hybrid(edge, cloud, hybrid, "fig3_hybrid_comparison.png")
    print(f"\nFigures and CSV files written to ./{OUT_DIR}/")


if __name__ == "__main__":
    main()
