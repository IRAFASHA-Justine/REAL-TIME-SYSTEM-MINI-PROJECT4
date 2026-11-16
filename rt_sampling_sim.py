# rt_sampling_sim.py -- Mini Project 4.1
# Simulating sampling jitter and processing overruns in a fixed-period
# sensing-and-control loop (Section 4.1 timing budget, Section 4.5 scan cycle).
#
# Pure Python + matplotlib. Run: python rt_sampling_sim.py
# Outputs: figures (PNG) and sweep_results.csv in ./output_4_1/

import bisect
import csv
import math
import os
import random
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------- parameters
AMPL = 1.0            # signal amplitude
SIG_PERIOD = 2.0      # period of the physical signal (s)
THRESHOLD = 0.5       # threshold controller trips on upward crossing
SETPOINT = 0.0        # P-controller setpoint
KP = 1.0              # P-controller gain
T_NOMINAL = 0.05      # nominal sampling period (s) -> 20 Hz
DURATION = 20.0       # run length (s)
C_TYPICAL = (0.002, 0.015)   # normal computation time range (s)
C_OVERRUN = (0.06, 0.20)     # overrun computation time range (s), > T_NOMINAL
OUT_DIR = "output_4_1"


# ------------------------------------------------------------------- process
def true_signal(t):
    """Sine wave + small continuous 'noise' (deterministic, so that ideal and
    jittered runs observe exactly the same physical process)."""
    noise = 0.008 * math.sin(2 * math.pi * 7.3 * t) \
          + 0.004 * math.sin(2 * math.pi * 13.1 * t + 1.0)
    return AMPL * math.sin(2 * math.pi * t / SIG_PERIOD) + noise


# ---------------------------------------------------------------- simulation
def simulate(T=T_NOMINAL, duration=DURATION, J_abs=0.0, p_over=0.0, seed=1):
    """Run the ideal baseline and the jittered/overrun loop.

    Scan-cycle model (Section 4.5): cycle k is *released* at k*T + jitter_k.
    It starts at max(release, finish of previous cycle) -- so an overrunning
    cycle delays the next one -- then samples the signal and computes for c_k.
    The actuation command is issued when the computation finishes.
    """
    assert C_OVERRUN[0] > T, "overrun range must exceed T_nominal"
    rng_j = random.Random(seed)
    rng_c = random.Random(seed + 10_000)
    N = int(round(duration / T))

    # baseline: t_k = k*T, zero jitter, zero computation delay
    ideal_t = [k * T for k in range(N)]
    ideal_y = [true_signal(t) for t in ideal_t]

    jitter = [rng_j.uniform(-J_abs, J_abs) if J_abs > 0 else 0.0 for _ in range(N)]
    comp = [rng_c.uniform(*C_OVERRUN) if rng_c.random() < p_over
            else rng_c.uniform(*C_TYPICAL) for _ in range(N)]

    start, finish = [], []
    prev_finish = 0.0
    for k in range(N):
        release = max(0.0, k * T + jitter[k])
        s = max(release, prev_finish)          # overrun delays next cycle
        prev_finish = s + comp[k]
        start.append(s)
        finish.append(prev_finish)
    real_y = [true_signal(s) for s in start]
    overrun = [c > T for c in comp]

    return dict(T=T, duration=duration, N=N, J_abs=J_abs, p_over=p_over,
                ideal_t=ideal_t, ideal_y=ideal_y, jitter=jitter, comp=comp,
                start=start, finish=finish, real_y=real_y, overrun=overrun)


# ---------------------------------------------------------------- controllers
def threshold_actuations(y, act_times):
    """Times at which a sample crosses THRESHOLD from below -> actuation."""
    return [act_times[k] for k in range(1, len(y))
            if y[k - 1] < THRESHOLD <= y[k]]


def zoh(times, values, grid):
    """Zero-order hold of (times, values) evaluated on grid (0 before first)."""
    out = []
    for g in grid:
        i = bisect.bisect_right(times, g) - 1
        out.append(values[i] if i >= 0 else 0.0)
    return out


def p_control_error(res):
    """RMS difference between the held P-controller output of the real loop and
    that of the ideal loop (output = KP*(SETPOINT - y), applied when issued)."""
    T = res["T"]
    grid = [i * T / 10 for i in range(int(res["duration"] / T * 10))]
    u_ideal = [KP * (SETPOINT - y) for y in res["ideal_y"]]
    u_real = [KP * (SETPOINT - y) for y in res["real_y"]]
    a = zoh(res["ideal_t"], u_ideal, grid)
    b = zoh(res["finish"], u_real, grid)
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)) / len(grid))


# -------------------------------------------------------------------- metrics
def match_delays(ideal_acts, real_acts):
    """For each ideal crossing find the nearest real one (within half a signal
    period); return (delays, number_missed)."""
    delays, missed = [], 0
    for ti in ideal_acts:
        cands = [tr for tr in real_acts if abs(tr - ti) <= SIG_PERIOD / 2]
        if cands:
            delays.append(min(cands, key=lambda tr: abs(tr - ti)) - ti)
        else:
            missed += 1
    return delays, missed


def metrics(res):
    T = res["T"]
    intervals = [b - a for a, b in zip(res["start"], res["start"][1:])]
    ideal_acts = threshold_actuations(res["ideal_y"], res["ideal_t"])
    real_acts = threshold_actuations(res["real_y"], res["finish"])
    delays, missed = match_delays(ideal_acts, real_acts)
    return dict(
        interval_mean=statistics.mean(intervals),
        interval_std=statistics.pstdev(intervals),
        max_overrun=max(0.0, max(res["comp"]) - T),
        pct_late=100 * sum(i > 1.001 * T for i in intervals) / len(intervals),
        pct_overrun=100 * sum(res["overrun"]) / res["N"],
        pct_gt_1p5T=100 * sum(i > 1.5 * T for i in intervals) / len(intervals),
        n_crossings=len(ideal_acts),
        mean_delay=statistics.mean(delays) if delays else float("nan"),
        max_delay=max(delays) if delays else float("nan"),
        missed=missed,
        p_rms=p_control_error(res),
        ideal_acts=ideal_acts, real_acts=real_acts, delays=delays,
    )


def averaged(T, J_abs, p_over, seeds=range(1, 21)):
    """Average metrics over several seeds (max_delay: worst case across seeds)."""
    ms = [metrics(simulate(T, DURATION, J_abs, p_over, s)) for s in seeds]
    out = {}
    for key in ("interval_mean", "interval_std", "pct_late", "pct_overrun",
                "pct_gt_1p5T", "mean_delay", "missed", "p_rms", "max_overrun"):
        vals = [m[key] for m in ms if not math.isnan(m[key])]
        out[key] = statistics.mean(vals) if vals else float("nan")
    d = [m["max_delay"] for m in ms if not math.isnan(m["max_delay"])]
    out["max_delay"] = max(d) if d else float("nan")
    return out


# ---------------------------------------------------------------------- plots
def fig1(res, m, name):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7))
    tmax = 8.0
    tt = [i * 0.002 for i in range(int(tmax / 0.002))]
    ax1.plot(tt, [true_signal(t) for t in tt], color="0.6", label="true signal")
    ax1.plot(res["ideal_t"], res["ideal_y"], "o", ms=4, mfc="none",
             color="tab:blue", label="ideal samples")
    ax1.plot(res["start"], res["real_y"], "x", ms=5, color="tab:red",
             label="jittered/overrun samples")
    ax1.axhline(THRESHOLD, ls=":", color="k", lw=0.8)
    ax1.set_xlim(0, tmax)
    ax1.set_ylabel("signal")
    ax1.set_title(f"Figure 1 - {name} (first {tmax:.0f} s shown)")
    ax1.legend(loc="upper right", fontsize=8)

    dev = [i - res["T"] for i in
           [b - a for a, b in zip(res["start"], res["start"][1:])]]
    ax2.plot(range(1, len(dev) + 1), dev, ".-", ms=3, lw=0.6, color="tab:red")
    ax2.axhline(0, color="k", lw=0.8)
    ax2.set_xlabel("cycle index")
    ax2.set_ylabel("interval - T_nominal (s)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"fig1_{name}.png"), dpi=130)
    plt.close(fig)


def fig2(res, m, name):
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.eventplot([m["ideal_acts"], m["real_acts"]], lineoffsets=[1, 0],
                 linelengths=0.6, colors=["tab:blue", "tab:red"])
    for ti, d in zip(m["ideal_acts"], m["delays"]):
        ax.annotate("", xy=(ti + d, 0.3), xytext=(ti, 0.7),
                    arrowprops=dict(arrowstyle="-", color="0.7", lw=0.8))
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["jittered/overrun", "ideal"])
    ax.set_xlabel("time (s)")
    ax.set_title(f"Figure 2 - actuation instants - {name}")
    txt = (f"mean delay = {m['mean_delay'] * 1e3:.1f} ms\n"
           f"max delay = {m['max_delay'] * 1e3:.1f} ms\n"
           f"missed crossings = {m['missed']}")
    ax.text(0.01, 0.5, txt, transform=ax.transAxes, fontsize=9,
            va="center", bbox=dict(fc="w", ec="0.6"))
    ax.set_ylim(-0.6, 1.6)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"fig2_{name}.png"), dpi=130)
    plt.close(fig)


# ----------------------------------------------------------------------- main
def fmt(x, scale=1.0, nd=2):
    return "nan" if (isinstance(x, float) and math.isnan(x)) else f"{x * scale:.{nd}f}"


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # sanity check (Expected Observations, bullet 1): J=0, p=0
    chk = metrics(simulate(J_abs=0.0, p_over=0.0))
    print(f"Sanity check J=0,p=0: mean delay = {chk['mean_delay'] * 1e3:.2f} ms "
          f"(just the normal compute time), missed = {chk['missed']}\n")

    # --- figures for three configurations (seed 1)
    figure_cfgs = {
        "jitter30pct": dict(J_abs=0.3 * T_NOMINAL, p_over=0.0),
        "overrun10pct": dict(J_abs=0.0, p_over=0.10),
        "combined": dict(J_abs=0.3 * T_NOMINAL, p_over=0.10),
    }
    for name, cfg in figure_cfgs.items():
        res = simulate(**cfg, seed=1)
        m = metrics(res)
        fig1(res, m, name)
        fig2(res, m, name)

    # --- sweep table (averaged over 20 seeds)
    sweep = [("baseline", 0.0, 0.0, T_NOMINAL)]
    sweep += [(f"jitter {int(f * 100)}%T", f * T_NOMINAL, 0.0, T_NOMINAL)
              for f in (0.1, 0.2, 0.3)]
    sweep += [(f"overrun p={p}", 0.0, p, T_NOMINAL) for p in (0.02, 0.05, 0.15)]
    sweep += [("jitter30%+over5%", 0.3 * T_NOMINAL, 0.05, T_NOMINAL)]
    # halved period, same absolute jitter / overrun magnitudes
    sweep += [("T/2 jitter 0.015s", 0.3 * T_NOMINAL, 0.0, T_NOMINAL / 2),
              ("T/2 overrun p=0.05", 0.0, 0.05, T_NOMINAL / 2)]

    header = ["config", "T(ms)", "J/T(%)", "int_mean(ms)", "int_std(ms)",
              "max_overrun(ms)", "%late", "%overrun_cycles", "%>1.5T",
              "mean_delay(ms)", "max_delay(ms)", "missed/run", "P_rms_err"]
    rows = []
    for name, J, p, T in sweep:
        a = averaged(T, J, p)
        rows.append([name, fmt(T, 1e3, 1), fmt(J / T, 100, 0),
                     fmt(a["interval_mean"], 1e3), fmt(a["interval_std"], 1e3),
                     fmt(a["max_overrun"], 1e3), fmt(a["pct_late"], 1, 1),
                     fmt(a["pct_overrun"], 1, 1), fmt(a["pct_gt_1p5T"], 1, 1),
                     fmt(a["mean_delay"], 1e3), fmt(a["max_delay"], 1e3),
                     fmt(a["missed"], 1, 2), fmt(a["p_rms"], 1, 4)])

    widths = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
    for r in [header] + rows:
        print("  ".join(str(c).ljust(w) for c, w in zip(r, widths)))
    with open(os.path.join(OUT_DIR, "sweep_results.csv"), "w", newline="") as f:
        csv.writer(f).writerows([header] + rows)
    print(f"\nFigures and sweep_results.csv written to ./{OUT_DIR}/")


if __name__ == "__main__":
    main()
