# -*- coding: utf-8 -*-
"""SOAC untuk project crashing pada data adiel (activity_data_v3.json, 25 task).

Ground truth biaya optimum dihitung dengan ILP (scipy.optimize.milp):
    min  sum_j c_j (dmax_j - d_j)
    s.t. s_j >= s_p + d_p            untuk tiap precedence (p -> j)
         s_j + d_j <= T_deadline     untuk tiap j
         dmin_j <= d_j <= dmax_j,  d_j integer,  s_j >= 0
Model tanpa kendala sumber daya bersifat linear, jadi ILP memberi optimum eksak.
"""

import json
import time
import numpy as np
from pathlib import Path  # Import Path
from scipy.optimize import milp, LinearConstraint, Bounds

from pysne.solver import solve_system
from project_crashing_problem import ProjectCrashingProblem

# DATA = "E:/p2ms/IMW26-AAA/adiel/data/activity_data_v3.json"
current_dir = Path(__file__).resolve().parent
DATA = current_dir.parent / "adiel" / "data" / "activity_data_v3.json"
DEADLINE = 243  # makespan normal = 249 -> perlu crash jalur kritis 6 hari


def load_tasks(path):
    raw = json.load(open(path))
    return [{"name": k,
             "predecessors": v["required_activities"],
             "d_min": v["activity_min_time"],
             "d_max": v["activity_normal_time"],
             "crash_cost": v["crash_cost"]} for k, v in raw.items()]


def ilp_ground_truth(problem):
    """Kembalikan (Z_optimal, d_optimal) via MILP."""
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
    d_opt = np.rint(res.x[:n]).astype(int)
    return problem.crash_cost(d_opt), d_opt


def main():
    problem = ProjectCrashingProblem(
        load_tasks(DATA), DEADLINE,
        unit_cube=True,
        params={
            "m_cluster": 2048, "k_cluster": 15, "gamma": 0.9,
            "r_cl": 0.95, "theta_cl": np.pi / 4,
            "sdoa_m": 128, "sdoa_k_max": 300,
            "sdoa_r": 0.97, "sdoa_theta": np.pi / 4,
            "delta": 0.4, "epsilon": 1e-9,
        },
    )
    print(f"{problem.name} | makespan normal={problem.makespan(problem.d_max)}, "
          f"min={problem.makespan(problem.d_min)}")

    z_star, d_star = ilp_ground_truth(problem)
    print(f"ILP ground truth: Z* = {z_star}")

    t0 = time.time()
    result = solve_system(problem, problem.get_info()[1], verbose=True)
    roots = result["roots"]
    print(f"Waktu SOAC: {time.time() - t0:.1f}s, cluster: {len(result['clusters'])}")

    if len(roots) == 0:
        print("SOAC tidak menemukan solusi feasible."); return
    costs = [problem.crash_cost(r) for r in roots]
    print(f"\nSOAC: {len(roots)} solusi, biaya terbaik = {min(costs)} "
          f"(gap thd ILP = {min(costs) - z_star})")
    print(problem.report(roots[:3]))
    if len(roots) > 3:
        print(f"... dan {len(roots) - 3} solusi optimum alternatif lainnya.")


if __name__ == "__main__":
    main()