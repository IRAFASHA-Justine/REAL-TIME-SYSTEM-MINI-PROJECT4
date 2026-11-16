"""rt_multirate_pipeline.py  --  Mini Project 4.2
Multi-rate environmental monitoring pipeline with timing-compliance reporting
(Section 4.1 timing budget per sensed quantity, Section 4.2 priority servicing).

One gateway processor services several sensor streams from ONE non-preemptive
scheduling loop. Pure Python + matplotlib.   Run:  python rt_multirate_pipeline.py
Outputs: figures (PNG) and summary_results.csv in ./output_4_2/

SCHEDULING RULES (applied only among streams that have a sample ready):
  FP  - fixed priority, shortest period first (fastest stream wins; a CAN-style
        static priority, Section 4.2). Ties: oldest release first.
  EDF - earliest deadline first, deadline = release + own period.
  RR  - round-robin: the ready stream that follows the last-served stream in
        a fixed cyclic order (no stream is favoured).

TIMING CLASSIFICATION (per required sample k of a stream with period P):
  release  r_k = k*P            deadline D_k = r_k + P
  on-time : read finished by D_k
  late    : read started before D_k but finished after it
            (period overrun = finish - D_k)
  missed  : read had not started when D_k passed; the stale sample is dropped
"""
import bisect
import csv
import heapq
import math
import os
import random
import statistics
from dataclasses import dataclass, field
from typing import Callable, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------- parameters
DURATION = 120.0                 # simulated run length (s)
P_TAIL = 0.05                    # probability a read hits the long tail
C_NORMAL = (0.010, 0.040)        # normal processing time per read (s)
C_TAIL = (0.150, 0.450)          # long-tail (overrun) processing time (s)
SEEDS = range(1, 21)             # seeds used for averaged results
PLOT_WINDOW = 40.0               # seconds shown in the signal plots
OUT_DIR = "output_4_2"


# --------------------------------------------------------------------- signals
# Time-compressed synthetic signals (smooth cycle + small disturbance).
def temp_sig(t):
    return 22 + 4 * math.sin(2 * math.pi * t / 10) + 0.15 * math.sin(2 * math.pi * t / 3.0)


def hum_sig(t):
    return 55 + 15 * math.sin(2 * math.pi * t / 15 + 1) + 1.0 * math.sin(2 * math.pi * t / 5)


def aqi_sig(t):
    return 80 + 40 * math.sin(2 * math.pi * t / 20 + 2) + 4 * math.sin(2 * math.pi * t / 8)


def vib_sig(t):
    return math.sin(2 * math.pi * t / 1.5) + 0.03 * math.sin(2 * math.pi * 5 * t)


_crossing_cache = {}


def true_crossings(name, fn, thr, t_end=DURATION + 1.0, dt=0.001):
    """Exact (interpolated) times at which the true signal crosses thr upward."""
    key = (name, thr)
    if key not in _crossing_cache:
        out, pt, py = [], 0.0, fn(0.0)
        for i in range(1, int(t_end / dt) + 1):
            t = i * dt
            y = fn(t)
            if py < thr <= y:
                out.append(pt + (thr - py) / (y - py) * dt)
            pt, py = t, y
        _crossing_cache[key] = out
    return _crossing_cache[key]


# --------------------------------------------------------------------- streams
@dataclass
class Stream:
    name: str
    unit: str
    period: float
    signal: Callable
    threshold: float
    crossings: list
    on_time: int = 0
    late: int = 0
    missed: int = 0
    overruns: list = field(default_factory=list)
    events: list = field(default_factory=list)   # (release, status, start, finish, y)
    alerts: list = field(default_factory=list)   # (detect_time, true_time or None)
    prev: Optional[tuple] = None                 # (start time, value) of last sample
    used: set = field(default_factory=set)

    @property
    def n_required(self):
        return math.ceil(DURATION / self.period - 1e-9)


STREAM_DEFS = [  # name, unit, period (s), signal, alert threshold
    ("temperature", "deg C", 0.5, temp_sig, 24.0),
    ("humidity", "% RH", 1.0, hum_sig, 65.0),
    ("air_quality", "AQI", 2.0, aqi_sig, 100.0),
    ("vibration", "g", 0.1, vib_sig, 0.6),          # extra fast stream (load B)
]


def make_streams(n_streams):
    return [Stream(n, u, p, f, th, true_crossings(n, f, th))
            for (n, u, p, f, th) in STREAM_DEFS[:n_streams]]


# ------------------------------------------------------------------- scheduler
def run(n_streams, rule, seed):
    streams = make_streams(n_streams)
    # per-sample processing times, fixed by (seed, stream, k) so every
    # scheduling rule faces exactly the same workload
    costs = []
    for i, s in enumerate(streams):
        rng = random.Random(seed * 1000 + i)
        costs.append([rng.uniform(*C_TAIL) if rng.random() < P_TAIL
                      else rng.uniform(*C_NORMAL) for _ in range(s.n_required)])

    release_heap = [(0.0, i, 0) for i in range(len(streams))]
    heapq.heapify(release_heap)
    ready, clock, last = [], 0.0, -1          # ready: (stream idx, k, release)

    while release_heap or ready:
        # move all releases that have happened into the ready list
        while release_heap and release_heap[0][0] <= clock:
            r, i, k = heapq.heappop(release_heap)
            ready.append((i, k, r))
            nxt = (k + 1) * streams[i].period
            if nxt < DURATION - 1e-9:
                heapq.heappush(release_heap, (nxt, i, k + 1))
        # drop stale samples whose deadline passed before they started -> missed
        keep = []
        for (i, k, r) in ready:
            s = streams[i]
            if r + s.period <= clock:
                s.missed += 1
                s.events.append((r, "missed", None, None, s.signal(r)))
            else:
                keep.append((i, k, r))
        ready = keep
        if not ready:                          # idle until next release
            if release_heap:
                clock = release_heap[0][0]
                continue
            break

        # ---- the documented scheduling rule
        if rule == "FP":
            pick = min(ready, key=lambda x: (streams[x[0]].period, x[2]))
        elif rule == "EDF":
            pick = min(ready, key=lambda x: (x[2] + streams[x[0]].period, x[0]))
        elif rule == "RR":
            pick = min(ready, key=lambda x: ((x[0] - last - 1) % len(streams), x[2]))
        else:
            raise ValueError(rule)
        ready.remove(pick)
        i, k, r = pick
        s = streams[i]

        # ---- service the read (non-preemptive)
        start = clock
        finish = start + costs[i][k]
        y = s.signal(start)
        deadline = r + s.period
        if finish <= deadline:
            s.on_time += 1
            status = "on-time"
        else:
            s.late += 1
            s.overruns.append(finish - deadline)
            status = "late"
        s.events.append((r, status, start, finish, y))

        # ---- alert rule: upward threshold crossing between consecutive samples
        if s.prev is not None and s.prev[1] < s.threshold <= y:
            j = bisect.bisect_right(s.crossings, s.prev[0])
            while j in s.used:
                j += 1
            if j < len(s.crossings) and s.crossings[j] <= start + 1e-9:
                s.used.add(j)
                s.alerts.append((finish, s.crossings[j]))
            else:
                s.alerts.append((finish, None))
        s.prev = (start, y)
        clock, last = finish, i
    return streams


# ---------------------------------------------------------------------- report
def summarize(s):
    lat = [d - t for d, t in s.alerts if t is not None]
    # only crossings that occur before the stream's last release can be detected
    n_true = sum(1 for c in s.crossings if c < (s.n_required - 1) * s.period)
    return dict(
        name=s.name, period=s.period, required=s.n_required,
        on_time=s.on_time, late=s.late, missed=s.missed,
        mean_over=statistics.mean(s.overruns) if s.overruns else 0.0,
        max_over=max(s.overruns) if s.overruns else 0.0,
        n_true=n_true, detected=len(lat),
        mean_lat=statistics.mean(lat) if lat else float("nan"),
        max_lat=max(lat) if lat else float("nan"))


def averaged(n_streams, rule):
    per_seed = [[summarize(s) for s in run(n_streams, rule, sd)] for sd in SEEDS]
    out = []
    for i in range(n_streams):
        rows = [ps[i] for ps in per_seed]
        req = rows[0]["required"]
        lats = [r["mean_lat"] for r in rows if not math.isnan(r["mean_lat"])]
        mx = [r["max_lat"] for r in rows if not math.isnan(r["max_lat"])]
        out.append(dict(
            name=rows[0]["name"], period=rows[0]["period"],
            pct_on=100 * statistics.mean(r["on_time"] for r in rows) / req,
            pct_late=100 * statistics.mean(r["late"] for r in rows) / req,
            pct_missed=100 * statistics.mean(r["missed"] for r in rows) / req,
            mean_over=statistics.mean(r["mean_over"] for r in rows),
            max_over=max(r["max_over"] for r in rows),
            pct_det=100 * statistics.mean(r["detected"] for r in rows) / rows[0]["n_true"],
            mean_lat=statistics.mean(lats) if lats else float("nan"),
            max_lat=max(mx) if mx else float("nan")))
    return out


def fmt(x, scale=1.0, nd=1):
    return "nan" if (isinstance(x, float) and math.isnan(x)) else f"{x * scale:.{nd}f}"


def print_table(header, rows):
    widths = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
    for r in [header] + rows:
        print("  ".join(str(c).ljust(w) for c, w in zip(r, widths)))


# ----------------------------------------------------------------------- plots
def plot_signals(streams, title, fname):
    fig, axes = plt.subplots(len(streams), 1, figsize=(11, 2.5 * len(streams)), sharex=True)
    tt = [i * 0.02 for i in range(int(PLOT_WINDOW / 0.02))]
    for ax, s in zip(axes, streams):
        ax.plot(tt, [s.signal(t) for t in tt], color="0.65", lw=1, label="true signal")
        ax.axhline(s.threshold, color="k", ls=":", lw=0.9, label="threshold")
        ok = [(e[2], e[4]) for e in s.events if e[1] == "on-time" and e[2] < PLOT_WINDOW]
        lt = [(e[2], e[4]) for e in s.events if e[1] == "late" and e[2] < PLOT_WINDOW]
        ms = [(e[0], e[4]) for e in s.events if e[1] == "missed" and e[0] < PLOT_WINDOW]
        if ok: ax.plot(*zip(*ok), "o", ms=3, color="tab:green", label="on-time")
        if lt: ax.plot(*zip(*lt), "^", ms=6, color="tab:orange", label="late")
        if ms: ax.plot(*zip(*ms), "x", ms=7, color="tab:red", label="missed")
        al = [(d, s.threshold) for d, _ in s.alerts if d < PLOT_WINDOW]
        if al: ax.plot(*zip(*al), "*", ms=11, color="tab:purple", label="alert (detected)")
        ax.set_ylabel(f"{s.name}\n({s.unit}, P={s.period:g}s)", fontsize=8)
        ax.set_xlim(0, PLOT_WINDOW)
    axes[0].legend(ncol=6, fontsize=7, loc="upper right")
    axes[-1].set_xlabel("time (s)")
    fig.suptitle(title + f"  (first {PLOT_WINDOW:.0f} s)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=120)
    plt.close(fig)


def plot_bars(streams, title, fname):
    fig, axes = plt.subplots(1, len(streams), figsize=(3.2 * len(streams), 3.4))
    for ax, s in zip(axes, streams):
        vals = [s.on_time, s.late, s.missed]
        bars = ax.bar(["on-time", "late", "missed"], vals,
                      color=["tab:green", "tab:orange", "tab:red"])
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v, str(v), ha="center", va="bottom", fontsize=8)
        ax.set_title(f"{s.name} (P={s.period:g}s)", fontsize=9)
        ax.set_ylim(0, max(vals) * 1.15 + 1)
        ax.tick_params(labelsize=8)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, fname), dpi=120)
    plt.close(fig)


# ------------------------------------------------------------------------ main
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    loads = {"A": 3, "B": 4}           # A: temp/humidity/air quality; B: + vibration
    rules = ["FP", "EDF", "RR"]

    # ---- per-stream compliance report (seed 1) + figures
    hdr = ["stream", "P(s)", "required", "on-time", "late", "missed",
           "mean_over(ms)", "max_over(ms)", "true_cross", "detected",
           "mean_alert_lat(ms)", "max_alert_lat(ms)"]
    for load, n in loads.items():
        for rule in rules:
            streams = run(n, rule, seed=1)
            print(f"\n=== Compliance report: load {load} ({n} streams), rule {rule}, seed 1 ===")
            rows = []
            for s in streams:
                m = summarize(s)
                rows.append([m["name"], f"{m['period']:g}", m["required"], m["on_time"],
                             m["late"], m["missed"], fmt(m["mean_over"], 1e3),
                             fmt(m["max_over"], 1e3), m["n_true"], m["detected"],
                             fmt(m["mean_lat"], 1e3), fmt(m["max_lat"], 1e3)])
            print_table(hdr, rows)
            title = f"Load {load} ({n} streams), {rule}"
            plot_signals(streams, title, f"signals_load{load}_{rule}.png")
            plot_bars(streams, title + " - on-time / late / missed samples",
                      f"bars_load{load}_{rule}.png")

    # ---- averaged comparison over 20 seeds
    hdr2 = ["load", "rule", "stream", "P(s)", "%on-time", "%late", "%missed",
            "mean_over(ms)", "max_over(ms)", "%alerts_detected",
            "mean_alert_lat(ms)", "max_alert_lat(ms)"]
    all_rows = []
    for load, n in loads.items():
        for rule in rules:
            for a in averaged(n, rule):
                all_rows.append([load, rule, a["name"], f"{a['period']:g}",
                                 fmt(a["pct_on"]), fmt(a["pct_late"]), fmt(a["pct_missed"]),
                                 fmt(a["mean_over"], 1e3), fmt(a["max_over"], 1e3),
                                 fmt(a["pct_det"]), fmt(a["mean_lat"], 1e3),
                                 fmt(a["max_lat"], 1e3)])
    print(f"\n=== Averaged over {len(SEEDS)} seeds (max columns = worst case) ===")
    print_table(hdr2, all_rows)
    with open(os.path.join(OUT_DIR, "summary_results.csv"), "w", newline="") as f:
        csv.writer(f).writerows([hdr2] + all_rows)
    print(f"\nFigures and summary_results.csv written to ./{OUT_DIR}/")


if __name__ == "__main__":
    main()

