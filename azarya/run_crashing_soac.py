# -*- coding: utf-8 -*-
"""Contoh penggunaan ProjectCrashingProblem dengan pipeline SOAC pysne,
divalidasi terhadap enumerasi brute-force (semua kombinasi durasi)."""

import itertools
import numpy as np

from pysne.solver import solve_system
from project_crashing_problem import ProjectCrashingProblem

# ------------------------------------------------------------------ #
# Data contoh (mengikuti pola tabel di slide: Bids, Grading, Site Work, ...)
# Rantai serial + dua task berbiaya crash sama -> optimum ganda (multimodal)
# ------------------------------------------------------------------ #
TASKS = [
    {"name": "Bids & Contracts",  "predecessors": [],                    "d_min": 7, "d_max": 10, "crash_cost": 60},
    {"name": "Grading & Permits", "predecessors": ["Bids & Contracts"],  "d_min": 7, "d_max": 10, "crash_cost": 70},
    {"name": "Site Work",         "predecessors": ["Grading & Permits"], "d_min": 5, "d_max": 7,  "crash_cost": 30},
    {"name": "Foundation",        "predecessors": ["Site Work"],         "d_min": 8, "d_max": 12, "crash_cost": 40},
    {"name": "Finishes",          "predecessors": ["Foundation"],        "d_min": 6, "d_max": 9,  "crash_cost": 40},
]
DEADLINE = 42  # makespan normal = 48 -> wajib crash 6 hari


def brute_force(problem):
    """Enumerasi seluruh grid durasi integer sebagai ground truth."""
    ranges = [range(lo, hi + 1) for lo, hi in problem.integer_domain]
    best_cost, best = float("inf"), []
    for combo in itertools.product(*ranges):
        d = np.array(combo)
        if problem.makespan(d) > problem.deadline:
            continue
        z = problem.crash_cost(d)
        if z < best_cost - 1e-9:
            best_cost, best = z, [combo]
        elif abs(z - best_cost) <= 1e-9:
            best.append(combo)
    return best_cost, sorted(best)


def main():
    problem = ProjectCrashingProblem(
        TASKS, DEADLINE,
        unit_cube=True,           # spiral di [0,1]^n_task, dekode di fitness
        params={
            "m_cluster": 512,     # titik Sobol fase clustering
            "k_cluster": 15,      # iterasi pembaruan cluster
            "gamma": 0.85,        # cutoff relatif thd F_best
            "r_cl": 0.95, "theta_cl": np.pi / 4,
            "sdoa_m": 64,         # titik Sobol per cluster (fase optimisasi)
            "sdoa_k_max": 150,
            "sdoa_r": 0.95, "sdoa_theta": np.pi / 4,
            "delta": 0.4, "epsilon": 1e-7,
        },
    )

    result = solve_system(problem, problem.get_info()[1], verbose=True)
    roots = result["roots"]

    print("\n=== Hasil SOAC ===")
    print(problem.report(roots))

    gt_cost, gt_sols = brute_force(problem)
    found = sorted(tuple(int(v) for v in r) for r in roots)
    print(f"\n=== Validasi brute force ===")
    print(f"Biaya minimum ground truth : {gt_cost}")
    print(f"Jumlah optimum ground truth: {len(gt_sols)} -> {gt_sols}")
    print(f"Jumlah optimum SOAC        : {len(found)} -> {found}")
    print(f"SEMUA optimum ditemukan    : {set(gt_sols) == set(found)}")


if __name__ == "__main__":
    main()