# -*- coding: utf-8 -*-
"""
run_sweep_tuner.py
==================
Sweep SOAC (integer) untuk setiap feasible deadline T (dari T_normal turun ke
T_min), dengan **grid search parameter otomatis** per deadline.

Untuk setiap T:
  1. Hitung ILP ground truth Z*
  2. Coba beberapa konfigurasi parameter (dari cepat ke berat)
  3. Jika gap=0, langsung pakai konfigurasi itu (early stopping)
  4. Jika tidak ada yang gap=0, pakai konfigurasi dengan gap terkecil
  5. Simpan output: summary.json, dist.png, gantt_*.png

Output:
  azarya/fix/outputs/sweep/
    T_<T>/
      summary.json
      dist.png
      gantt_sol<NNN>_ms<MS>.png
    sweep_summary.json       -- ringkasan global semua deadline
    tuning_report.json       -- laporan lengkap grid search per deadline
"""

import json
import os
import sys
import time
from collections import Counter
from itertools import product
from pathlib import Path

import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from pysne.solver import solve_system
from project_crashing_problem_new import ProjectCrashingProblem

# =====================================================================
# KONFIGURASI
# =====================================================================

current_dir = Path(__file__).resolve().parent
DATA = current_dir.parent.parent / "adiel" / "data" / "activity_data_v3.json"
# Output root
OUT_ROOT = current_dir / "outputs" / "sweep"

# =====================================================================
# GRID SEARCH: Parameter configurations
# =====================================================================
# Tiered configs: dari ringan (cepat) ke berat (lambat tapi akurat).
# Untuk setiap T, dicoba secara berurutan. Berhenti begitu gap=0.

# Parameter yang di-tune dan grid values-nya:
GRID = {
    "m_cluster":  [4096, 16384, 32768],
    "gamma":      [0.85, 0.9],
    "sdoa_r":     [0.97, 0.99],
    "sdoa_theta": [np.pi / 4, np.pi / 16],
}

# Parameter tetap (tidak di-tune):
FIXED_PARAMS = {
    "k_cluster": 100,
    "r_cl": 0.95,
    "theta_cl": np.pi / 4,
    "sdoa_m": 1024,
    "sdoa_k_max": 1000,
    "delta": 0.00001,
    "epsilon": 1e-9,
}


def build_param_configs():
    """Generate semua kombinasi parameter dari grid, urut dari ringan ke berat.

    Diurutkan berdasarkan m_cluster (proxy utama untuk waktu komputasi).
    """
    keys = list(GRID.keys())
    configs = []
    for vals in product(*[GRID[k] for k in keys]):
        cfg = dict(zip(keys, vals))
        cfg.update(FIXED_PARAMS)
        configs.append(cfg)
    # Urutkan dari ringan ke berat (m_cluster terkecil dulu)
    configs.sort(key=lambda c: (c["m_cluster"], c["gamma"]))
    return configs


PARAM_CONFIGS = build_param_configs()


def config_label(cfg):
    """Label singkat untuk konfigurasi parameter."""
    return (f"m={cfg['m_cluster']}, g={cfg['gamma']}, "
            f"sr={cfg['sdoa_r']}, st={cfg['sdoa_theta']:.4f}")


# =====================================================================
# DATA LOADING
# =====================================================================

def load_tasks(path):
    """Muat task dari JSON (format activity_data_v3.json)."""
    raw = json.load(open(path))
    return [{"name": k,
             "predecessors": v["required_activities"],
             "d_min": v["activity_min_time"],
             "d_max": v["activity_normal_time"],
             "crash_cost": v["crash_cost"]} for k, v in raw.items()]


# =====================================================================
# ILP GROUND TRUTH
# =====================================================================

def ilp_ground_truth(problem):
    """Kembalikan (Z_optimal, d_optimal) via MILP -- solusi integer eksak."""
    n = problem.n_tasks
    # Variabel: [d_0..d_{n-1}, s_0..s_{n-1}]
    c_obj = np.concatenate([-problem.c, np.zeros(n)])  # min -sum c_j d_j
    A_rows, lb, ub = [], [], []
    for j, preds in enumerate(problem.pred_idx):
        for p in preds:                       # s_j - s_p - d_p >= 0
            row = np.zeros(2 * n)
            row[n + j], row[n + p], row[p] = 1, -1, -1
            A_rows.append(row); lb.append(0); ub.append(np.inf)
    for j in range(n):                        # s_j + d_j <= T
        row = np.zeros(2 * n)
        row[n + j], row[j] = 1, 1
        A_rows.append(row); lb.append(-np.inf); ub.append(problem.deadline)
    cons = LinearConstraint(np.array(A_rows), lb, ub)
    bounds = Bounds(np.concatenate([problem.d_min, np.zeros(n)]),
                    np.concatenate([problem.d_max, np.full(n, np.inf)]))
    integrality = np.concatenate([np.ones(n), np.zeros(n)])
    res = milp(c=c_obj, constraints=cons, bounds=bounds, integrality=integrality)
    if not res.success:
        return None, None
    d_opt = np.rint(res.x[:n]).astype(int)
    return problem.crash_cost(d_opt), d_opt


# =====================================================================
# VISUALISASI -- GANTT CHART
# =====================================================================

def plot_gantt_comparison(problem, d_opt, output_path, title_suffix=""):
    """Gambar Gantt baseline (d = d_max) vs jadwal hasil crashing d_opt."""
    d_opt = np.asarray(d_opt, dtype=int)
    d_base = problem.d_max
    s_b, e_b = problem.schedule(d_base)
    s_o, e_o = problem.schedule(d_opt)
    names = problem.task_names
    n = problem.n_tasks

    order_b = sorted(range(n), key=lambda j: (s_b[j], e_b[j], names[j]))
    order_o = sorted(range(n), key=lambda j: (s_o[j], e_o[j], names[j]))

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(18, max(8, 0.45 * n)), sharey=False)

    def draw(ax, order, s, e, colors, title, end_line=None):
        for idx, j in enumerate(order):
            dur = int(e[j] - s[j])
            ax.barh(idx, dur, left=s[j], height=0.6, color=colors[j],
                    edgecolor="black", linewidth=0.5)
            label = str(dur)
            if dur > 2:
                ax.text(s[j] + dur / 2, idx, label, va="center", ha="center",
                        color="white", fontsize=8, weight="bold")
            else:
                ax.text(e[j] + 0.5, idx, label, va="center", ha="left",
                        color="black", fontsize=8)
        ax.set_title(title, fontsize=14, pad=15)
        ax.set_xlabel("Project Day", fontsize=12)
        ax.set_yticks(range(len(order)))
        ax.set_yticklabels([names[j] for j in order], fontsize=9)
        ax.invert_yaxis()
        ax.grid(axis="x", linestyle="--", alpha=0.5)
        if end_line is not None:
            ax.axvline(x=end_line, color="red", linestyle="--", linewidth=1.5)

    # Panel kiri: baseline (semua biru)
    draw(ax1, order_b, s_b, e_b, ["#3498db"] * n,
         f"Original Schedule (Baseline) -- makespan {e_b.max()}")

    # Panel kanan: optimized (merah jika di-crash)
    crashed = d_opt < d_base
    colors_o = ["#e74c3c" if crashed[j] else "#3498db" for j in range(n)]
    z = problem.crash_cost(d_opt)
    draw(ax2, order_o, s_o, e_o, colors_o,
         f"SOAC Optimized{title_suffix} -- makespan {e_o.max()}, "
         f"crash cost = {z:.0f}",
         end_line=e_o.max())

    legend = [
        Patch(facecolor="#3498db", edgecolor="black", label="Normal (No Crash)"),
        Patch(facecolor="#e74c3c", edgecolor="black",
              label="Crashed (durasi < normal)"),
        Line2D([0], [0], color="red", linestyle="--", linewidth=1.5,
               label=f"Project End Date (Day {e_o.max()}) | "
                     f"Deadline T = {problem.deadline}"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=3, fontsize=10,
               frameon=True)
    fig.suptitle(f"Project Crashing via SOAC -- {problem.name}", fontsize=15)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


# =====================================================================
# VISUALISASI -- DISTRIBUSI MAKESPAN
# =====================================================================

def save_distribution_plot(T, cost, counts, num_solutions, out_path,
                           config_name=""):
    """Simpan bar chart distribusi makespan untuk deadline T tertentu."""
    plt.figure(figsize=(10, 6), dpi=150)

    x_vals = sorted(counts.keys())
    y_vals = [counts[x] for x in x_vals]

    base_color = "#1e82a8"
    match_color = "#e67e22"
    colors = [match_color if x <= T else base_color for x in x_vals]

    bars = plt.bar([str(x) for x in x_vals], y_vals, color=colors,
                   width=0.6, edgecolor="none", zorder=3)

    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0,
                 h + max(0.05, max(y_vals) * 0.02) if y_vals else 0.05,
                 f"{int(h)}", ha="center", va="bottom",
                 fontsize=9, fontweight="bold", color="#2c3e50")

    title = (f"Makespan Distribution -- Deadline T = {T} days  |  "
             f"Min Cost = {cost:.0f}  |  {num_solutions} solutions")
    if config_name:
        title += f"\n({config_name})"
    plt.title(title, fontsize=12, fontweight="bold", pad=14, color="#2c3e50")
    plt.xlabel("Project Makespan (Days)", fontsize=11, fontweight="bold",
               labelpad=8, color="#2c3e50")
    plt.ylabel("Number of Solutions", fontsize=11, fontweight="bold",
               labelpad=8, color="#2c3e50")
    if y_vals:
        plt.ylim(0, max(y_vals) + max(1, max(y_vals) * 0.15))
    plt.grid(axis="y", linestyle="--", alpha=0.7, zorder=0)

    meeting_count = sum(counts.get(x, 0) for x in x_vals if x <= T)
    if meeting_count > 0:
        plt.figtext(0.02, 0.02,
                    f"* {meeting_count} solution(s) meet deadline T={T}",
                    fontsize=10, fontweight="bold", color=match_color)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


# =====================================================================
# GRID SEARCH PER DEADLINE
# =====================================================================

def run_soac_with_config(tasks, T, cfg, z_star):
    """Jalankan SOAC dengan konfigurasi tertentu, kembalikan hasil."""
    problem = ProjectCrashingProblem(tasks, deadline=T, params=cfg)
    t0 = time.time()
    result = solve_system(problem, problem.get_info()[1], verbose=False)
    elapsed = time.time() - t0
    roots = result["roots"]

    if len(roots) == 0:
        return {
            "config": cfg,
            "soac_best": None,
            "gap": None,
            "num_solutions": 0,
            "roots": [],
            "costs": [],
            "makespans": [],
            "elapsed": elapsed,
            "problem": problem,
        }

    costs = [problem.crash_cost(r) for r in roots]
    makespans = [problem.makespan(r) for r in roots]
    soac_best = min(costs)
    gap = soac_best - z_star

    return {
        "config": cfg,
        "soac_best": soac_best,
        "gap": gap,
        "num_solutions": len(roots),
        "roots": roots,
        "costs": costs,
        "makespans": makespans,
        "elapsed": elapsed,
        "problem": problem,
    }


def grid_search_for_deadline(tasks, T, z_star):
    """Coba semua konfigurasi parameter, berhenti jika gap=0."""
    all_trials = []
    best_result = None
    best_gap = float("inf")

    for i, cfg in enumerate(PARAM_CONFIGS):
        label = config_label(cfg)
        print(f"    Config {i+1}/{len(PARAM_CONFIGS)}: {label} ...", end=" ",
              flush=True)

        trial = run_soac_with_config(tasks, T, cfg, z_star)
        trial["config_idx"] = i
        trial["config_label"] = label
        all_trials.append(trial)

        if trial["gap"] is not None:
            print(f"Z={trial['soac_best']:.0f}, gap={trial['gap']:.0f}, "
                  f"{trial['num_solutions']} sol, {trial['elapsed']:.1f}s")

            if trial["gap"] < best_gap:
                best_gap = trial["gap"]
                best_result = trial

            # Early stopping: gap=0 berarti optimal ditemukan
            if trial["gap"] <= 0:
                print(f"    >> GAP=0! Optimal ditemukan, skip sisa configs.")
                break
        else:
            print(f"no feasible solution, {trial['elapsed']:.1f}s")

    return best_result, all_trials


# =====================================================================
# MAIN SWEEP
# =====================================================================

def run_sweep():
    print("=" * 80)
    print("SOAC PROJECT CRASHING SWEEP + GRID SEARCH TUNER (Integer)")
    print("=" * 80)

    tasks = load_tasks(DATA)

    # Tentukan T_normal dan T_min
    temp_params = dict(FIXED_PARAMS)
    temp_params["m_cluster"] = 4096  # minimal, hanya untuk makespan calc
    temp_problem = ProjectCrashingProblem(tasks, deadline=9999, params=temp_params)
    T_normal = temp_problem.makespan(temp_problem.d_max)
    T_min_abs = temp_problem.makespan(temp_problem.d_min)
    T_min = max(T_min_abs, T_normal - 17)  # 249 - 17 = 232, sesuai Adiel

    n_configs = len(PARAM_CONFIGS)
    print(f"\nMakespan normal (tanpa crashing) : T_normal = {T_normal}")
    print(f"Makespan minimum (crash penuh)   : T_min    = {T_min}")
    print(f"Deadline range: T = {T_normal} -> {T_min}  "
          f"({T_normal - T_min + 1} values)")
    print(f"Parameter grid: {n_configs} configurations")
    print(f"Grid dimensions: {', '.join(f'{k}={len(v)}' for k,v in GRID.items())}")
    print()

    os.makedirs(OUT_ROOT, exist_ok=True)
    sweep_results = []
    tuning_report = []  # detail grid search per T

    for T in range(T_normal, T_min - 1, -1):
        print("-" * 70)
        print(f"[T = {T}]  Mulai grid search...")
        t_iter = time.time()

        # 1. ILP ground truth
        temp_problem_T = ProjectCrashingProblem(
            tasks, deadline=T, params=temp_params)
        z_star, d_star = ilp_ground_truth(temp_problem_T)
        if z_star is None:
            print(f"  X ILP infeasible untuk T={T}, skip.")
            sweep_results.append({"T": T, "status": "INFEASIBLE"})
            tuning_report.append({"T": T, "status": "INFEASIBLE", "trials": []})
            continue
        print(f"  ILP ground truth: Z* = {z_star:.0f}")

        # 2. Kasus trivial: Z*=0 (tidak perlu crashing)
        if z_star == 0:
            print(f"  Z*=0: tidak perlu crashing, skip SOAC.")
            out_dir = OUT_ROOT / f"T_{T}"
            os.makedirs(out_dir, exist_ok=True)
            # Simpan distribution trivial
            counts = Counter({T_normal: 1})
            save_distribution_plot(T, 0, counts, 1, str(out_dir / "dist.png"),
                                   "No crashing needed")
            summary = {
                "T": T,
                "status": "OK",
                "ilp_min_cost": 0.0,
                "soac_min_cost": 0.0,
                "gap": 0.0,
                "num_solutions": 1,
                "makespan_frequencies": {str(T_normal): 1},
                "num_meeting_deadline": 0,
                "gantt_files": [],
                "best_config": "N/A (no crashing needed)",
                "elapsed_seconds": round(time.time() - t_iter, 2),
            }
            with open(out_dir / "summary.json", "w") as f:
                json.dump(summary, f, indent=2)
            sweep_results.append(summary)
            tuning_report.append({
                "T": T, "status": "TRIVIAL",
                "z_star": 0, "trials": [],
            })
            print(f"  Done T={T} (trivial)")
            continue

        # 3. Grid search: coba semua konfigurasi
        best, all_trials = grid_search_for_deadline(tasks, T, z_star)

        # Log trial ke tuning report (tanpa roots/problem yang besar)
        trial_log = []
        for tr in all_trials:
            trial_log.append({
                "config_idx": tr["config_idx"],
                "config_label": tr["config_label"],
                "config": {k: (float(v) if isinstance(v, (int, float, np.floating))
                               else str(v))
                           for k, v in tr["config"].items()},
                "soac_best": tr["soac_best"],
                "gap": tr["gap"],
                "num_solutions": tr["num_solutions"],
                "elapsed": round(tr["elapsed"], 2),
            })

        if best is None or best["num_solutions"] == 0:
            print(f"  X Semua config gagal untuk T={T}")
            out_dir = OUT_ROOT / f"T_{T}"
            os.makedirs(out_dir, exist_ok=True)
            summary = {
                "T": T,
                "status": "NO_SOAC_SOLUTION",
                "ilp_min_cost": z_star,
                "soac_min_cost": None,
                "gap": None,
                "num_solutions": 0,
                "best_config": None,
                "configs_tried": len(all_trials),
                "elapsed_seconds": round(time.time() - t_iter, 2),
            }
            with open(out_dir / "summary.json", "w") as f:
                json.dump(summary, f, indent=2)
            sweep_results.append(summary)
            tuning_report.append({
                "T": T, "status": "NO_SOLUTION",
                "z_star": z_star, "trials": trial_log,
            })
            continue

        # 4. Gunakan hasil terbaik
        problem = best["problem"]
        roots = best["roots"]
        costs = best["costs"]
        makespans_list = best["makespans"]
        soac_best = best["soac_best"]
        gap = best["gap"]

        print(f"  >> Best config: {best['config_label']}")
        print(f"  >> SOAC best: Z={soac_best:.0f}, gap={gap:.0f}, "
              f"{best['num_solutions']} sol")

        # 5. Distribusi makespan
        counts = Counter(makespans_list)
        print(f"  Distribusi makespan:")
        for ms in sorted(counts.keys()):
            marker = " <- DEADLINE MET" if ms <= T else ""
            print(f"    makespan {ms}: {counts[ms]} solusi{marker}")

        # 6. Simpan output
        out_dir = OUT_ROOT / f"T_{T}"
        os.makedirs(out_dir, exist_ok=True)

        # 6a. Distribution plot
        dist_path = out_dir / "dist.png"
        save_distribution_plot(T, soac_best, counts, len(roots),
                               str(dist_path), best["config_label"])
        print(f"  [dist] Saved -> {dist_path}")

        # 6b. Gantt charts
        meeting = [(k, roots[k], costs[k], makespans_list[k])
                   for k in range(len(roots))
                   if makespans_list[k] <= T]
        gantt_files = []

        print(f"  Solusi meeting deadline (makespan <= {T}): {len(meeting)}")
        for rank, (k, root, cost, ms) in enumerate(meeting, start=1):
            d = np.asarray(root, dtype=int)
            gantt_name = f"gantt_sol{rank:03d}_ms{ms}.png"
            gantt_path = out_dir / gantt_name
            try:
                plot_gantt_comparison(
                    problem, d, str(gantt_path),
                    title_suffix=f" (solusi #{rank} dari {len(meeting)})")
                gantt_files.append(gantt_name)
                print(f"    [gantt] Saved sol {rank} (ms={ms}, Z={cost:.0f}) "
                      f"-> {gantt_path}")
            except Exception as exc:
                print(f"    [gantt] WARNING: Gagal sol {rank}: {exc}")

        # 6c. Summary JSON
        elapsed = round(time.time() - t_iter, 2)
        summary = {
            "T": T,
            "status": "OK",
            "ilp_min_cost": z_star,
            "soac_min_cost": soac_best,
            "gap": gap,
            "num_solutions": len(roots),
            "makespan_frequencies": {str(k): v for k, v in sorted(counts.items())},
            "num_meeting_deadline": len(meeting),
            "gantt_files": gantt_files,
            "best_config": best["config_label"],
            "best_config_params": {
                k: (float(v) if isinstance(v, (int, float, np.floating))
                    else str(v))
                for k, v in best["config"].items()
            },
            "configs_tried": len(all_trials),
            "elapsed_seconds": elapsed,
        }
        with open(out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        sweep_results.append(summary)
        tuning_report.append({
            "T": T, "status": "OK", "z_star": z_star,
            "best_config_idx": best["config_idx"],
            "best_config_label": best["config_label"],
            "best_gap": gap,
            "trials": trial_log,
        })
        print(f"  Done T={T} in {elapsed}s "
              f"({len(all_trials)}/{n_configs} configs tried)")

    # ================================================================
    # FINAL SUMMARY
    # ================================================================
    print("\n" + "=" * 80)
    print("SWEEP COMPLETE")
    print("=" * 80)
    header = (f"{'T':>6}  {'ILP Z*':>8}  {'SOAC Z':>8}  {'Gap':>5}  "
              f"{'#Sol':>5}  {'Meet':>5}  {'Tried':>6}  {'Time':>7}  "
              f"Best Config")
    print(header)
    print("-" * len(header) + "-" * 30)
    for entry in sweep_results:
        if entry.get("status") == "INFEASIBLE":
            print(f"{entry['T']:>6}  {'---':>8}  {'INFEAS':>8}")
        elif entry.get("status") == "NO_SOAC_SOLUTION":
            print(f"{entry['T']:>6}  {entry['ilp_min_cost']:>8.0f}  "
                  f"{'NOSOL':>8}  {'---':>5}  {0:>5}  {'---':>5}  "
                  f"{entry.get('configs_tried',0):>6}  "
                  f"{entry['elapsed_seconds']:>7.1f}")
        else:
            bc = entry.get("best_config", "")
            ct = entry.get("configs_tried", "")
            print(f"{entry['T']:>6}  {entry['ilp_min_cost']:>8.0f}  "
                  f"{entry['soac_min_cost']:>8.0f}  {entry['gap']:>5.0f}  "
                  f"{entry['num_solutions']:>5}  "
                  f"{entry['num_meeting_deadline']:>5}  "
                  f"{ct:>6}  "
                  f"{entry['elapsed_seconds']:>7.1f}  {bc}")

    # Simpan global summary
    global_path = OUT_ROOT / "sweep_summary.json"
    with open(global_path, "w") as f:
        json.dump({
            "T_normal": T_normal,
            "T_min": T_min,
            "grid": {k: [float(v) if isinstance(v, (int, float, np.floating))
                         else str(v) for v in vals]
                     for k, vals in GRID.items()},
            "fixed_params": {k: (float(v) if isinstance(v, (int, float, np.floating))
                                 else str(v))
                             for k, v in FIXED_PARAMS.items()},
            "num_configs": n_configs,
            "results": sweep_results,
        }, f, indent=2)
    print(f"\nGlobal summary saved -> {global_path}")

    # Simpan tuning report
    tuning_path = OUT_ROOT / "tuning_report.json"
    with open(tuning_path, "w") as f:
        json.dump(tuning_report, f, indent=2)
    print(f"Tuning report saved  -> {tuning_path}")
    print(f"All outputs in       -> {OUT_ROOT}")


if __name__ == "__main__":
    run_sweep()
