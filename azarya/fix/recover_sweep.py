# -*- coding: utf-8 -*-
"""
recover_sweep.py
================
Regenerate sweep output files using the best config per T from previous run.
Only runs 1 SOAC per T (not full grid search), so much faster.
"""

import json
import os
import time
from collections import Counter
from pathlib import Path
from math import pi

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_sweep_tuner import (
    load_tasks, ilp_ground_truth, run_soac_with_config,
    plot_gantt_comparison, save_distribution_plot,
    DATA, FIXED_PARAMS,
)
from project_crashing_problem_new import ProjectCrashingProblem

OUT_ROOT = Path(__file__).resolve().parent / "outputs" / "sweep"

# Best configs per T dari run sebelumnya
BEST_CONFIGS = {
    # T: {tuned params only}
    249: None,  # trivial, Z*=0
    248: {"m_cluster": 4096, "gamma": 0.85, "sdoa_r": 0.97, "sdoa_theta": pi/4},
    247: {"m_cluster": 4096, "gamma": 0.85, "sdoa_r": 0.97, "sdoa_theta": pi/4},
    246: {"m_cluster": 4096, "gamma": 0.85, "sdoa_r": 0.97, "sdoa_theta": pi/16},
    245: {"m_cluster": 4096, "gamma": 0.85, "sdoa_r": 0.97, "sdoa_theta": pi/4},
    244: {"m_cluster": 4096, "gamma": 0.85, "sdoa_r": 0.99, "sdoa_theta": pi/16},
    243: {"m_cluster": 4096, "gamma": 0.85, "sdoa_r": 0.99, "sdoa_theta": pi/16},
    242: {"m_cluster": 4096, "gamma": 0.9,  "sdoa_r": 0.99, "sdoa_theta": pi/16},
    241: {"m_cluster": 4096, "gamma": 0.85, "sdoa_r": 0.99, "sdoa_theta": pi/16},
    240: {"m_cluster": 4096, "gamma": 0.9,  "sdoa_r": 0.99, "sdoa_theta": pi/16},
    239: {"m_cluster": 4096, "gamma": 0.85, "sdoa_r": 0.97, "sdoa_theta": pi/16},
    238: {"m_cluster": 4096, "gamma": 0.85, "sdoa_r": 0.97, "sdoa_theta": pi/16},
    237: {"m_cluster": 4096, "gamma": 0.85, "sdoa_r": 0.97, "sdoa_theta": pi/4},
    236: {"m_cluster": 4096, "gamma": 0.9,  "sdoa_r": 0.97, "sdoa_theta": pi/4},
    235: {"m_cluster": 4096, "gamma": 0.85, "sdoa_r": 0.99, "sdoa_theta": pi/4},
    234: {"m_cluster": 4096, "gamma": 0.85, "sdoa_r": 0.97, "sdoa_theta": pi/4},
    233: {"m_cluster": 4096, "gamma": 0.85, "sdoa_r": 0.97, "sdoa_theta": pi/4},
    232: {"m_cluster": 4096, "gamma": 0.85, "sdoa_r": 0.97, "sdoa_theta": pi/4},
}


def config_label(cfg):
    return (f"m={cfg['m_cluster']}, g={cfg['gamma']}, "
            f"sr={cfg['sdoa_r']}, st={cfg['sdoa_theta']:.4f}")


def main():
    print("=" * 70)
    print("RECOVERING SWEEP OUTPUTS (best config per T, 1 run each)")
    print("=" * 70)

    tasks = load_tasks(DATA)
    temp_params = dict(FIXED_PARAMS)
    temp_params["m_cluster"] = 4096
    temp_problem = ProjectCrashingProblem(tasks, deadline=9999, params=temp_params)
    T_normal = temp_problem.makespan(temp_problem.d_max)

    os.makedirs(OUT_ROOT, exist_ok=True)
    sweep_results = []

    for T in range(249, 231, -1):
        print("-" * 60)
        print(f"[T = {T}]")
        t0 = time.time()

        # ILP ground truth
        p_tmp = ProjectCrashingProblem(tasks, deadline=T, params=temp_params)
        z_star, d_star = ilp_ground_truth(p_tmp)

        out_dir = OUT_ROOT / f"T_{T}"
        os.makedirs(out_dir, exist_ok=True)

        # Trivial case
        if z_star == 0 or BEST_CONFIGS[T] is None:
            print(f"  Z*=0, trivial.")
            counts = Counter({T_normal: 1})
            save_distribution_plot(T, 0, counts, 1, str(out_dir / "dist.png"),
                                   "No crashing needed")
            summary = {
                "T": T, "status": "OK",
                "ilp_min_cost": 0.0, "soac_min_cost": 0.0, "gap": 0.0,
                "num_solutions": 1,
                "makespan_frequencies": {str(T_normal): 1},
                "num_meeting_deadline": 0, "gantt_files": [],
                "best_config": "N/A (no crashing needed)",
                "elapsed_seconds": round(time.time() - t0, 2),
            }
            with open(out_dir / "summary.json", "w") as f:
                json.dump(summary, f, indent=2)
            sweep_results.append(summary)
            continue

        # Build full config
        cfg = dict(FIXED_PARAMS)
        cfg.update(BEST_CONFIGS[T])
        label = config_label(cfg)
        print(f"  Z*={z_star:.0f}, config: {label}")

        # Run SOAC with the known best config
        print(f"  Running SOAC...", end=" ", flush=True)
        result = run_soac_with_config(tasks, T, cfg, z_star)
        print(f"Z={result['soac_best']}, gap={result['gap']}, "
              f"{result['num_solutions']} sol, {result['elapsed']:.1f}s")

        if result["num_solutions"] == 0:
            print(f"  WARNING: no solution found, saving empty.")
            summary = {
                "T": T, "status": "NO_SOAC_SOLUTION",
                "ilp_min_cost": z_star, "soac_min_cost": None, "gap": None,
                "num_solutions": 0, "best_config": label,
                "elapsed_seconds": round(time.time() - t0, 2),
            }
            with open(out_dir / "summary.json", "w") as f:
                json.dump(summary, f, indent=2)
            sweep_results.append(summary)
            continue

        problem = result["problem"]
        roots = result["roots"]
        costs = result["costs"]
        makespans_list = result["makespans"]

        # Distribution plot
        counts = Counter(makespans_list)
        save_distribution_plot(T, result["soac_best"], counts,
                               len(roots), str(out_dir / "dist.png"), label)

        # Gantt charts
        meeting = [(k, roots[k], costs[k], makespans_list[k])
                   for k in range(len(roots))
                   if makespans_list[k] <= T]
        gantt_files = []
        for rank, (k, root, cost, ms) in enumerate(meeting, start=1):
            d = np.asarray(root, dtype=int)
            gantt_name = f"gantt_sol{rank:03d}_ms{ms}.png"
            gantt_path = out_dir / gantt_name
            try:
                plot_gantt_comparison(
                    problem, d, str(gantt_path),
                    title_suffix=f" (solusi #{rank} dari {len(meeting)})")
                gantt_files.append(gantt_name)
                print(f"    [gantt] sol {rank} (ms={ms}) saved")
            except Exception as exc:
                print(f"    [gantt] FAILED: {exc}")

        # Summary JSON
        elapsed = round(time.time() - t0, 2)
        summary = {
            "T": T, "status": "OK",
            "ilp_min_cost": z_star,
            "soac_min_cost": result["soac_best"],
            "gap": result["gap"],
            "num_solutions": len(roots),
            "makespan_frequencies": {str(k): v for k, v in sorted(counts.items())},
            "num_meeting_deadline": len(meeting),
            "gantt_files": gantt_files,
            "best_config": label,
            "best_config_params": {
                k2: (float(v2) if isinstance(v2, (int, float, np.floating))
                     else str(v2))
                for k2, v2 in cfg.items()
            },
            "elapsed_seconds": elapsed,
        }
        with open(out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        sweep_results.append(summary)
        print(f"  Done T={T} in {elapsed}s")

    # Global summary
    global_path = OUT_ROOT / "sweep_summary.json"
    with open(global_path, "w") as f:
        json.dump({"T_normal": 249, "T_min": 232, "results": sweep_results},
                  f, indent=2)
    print(f"\nSaved -> {global_path}")
    print(f"All outputs -> {OUT_ROOT}")


if __name__ == "__main__":
    main()
