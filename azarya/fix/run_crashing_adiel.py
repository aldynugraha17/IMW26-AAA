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
from project_crashing_problem_new import ProjectCrashingProblem

# DATA = "/home/claude/IMW26-AAA/adiel/data/activity_data_v3.json"
# DATA = "E:/p2ms/IMW26-AAA/adiel/data/activity_data_v3.json"
current_dir = Path(__file__).resolve().parent
DATA = current_dir.parent.parent / "adiel" / "data" / "activity_data_v3.json"
DEADLINE = 241  # makespan normal = 249 -> perlu crash jalur kritis 6 hari


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


PARAMS = dict(
    m_cluster=4096 * 4, k_cluster=100, gamma=0.85,
    r_cl=0.95, theta_cl=np.pi / 4,
    sdoa_m=1024, sdoa_k_max=1000,
    sdoa_r=0.99, sdoa_theta=np.pi / 16,
    delta=0.00001, epsilon=1e-9,
)


def run_one(problem, z_star, label):
    t0 = time.time()
    print(f"Menjalankan SOAC dengan parameter: {PARAMS}")
    result = solve_system(problem, problem.get_info()[1], verbose=True)
    roots = result["roots"]
    print(f"Waktu SOAC: {time.time() - t0:.1f}s, cluster: {len(result['clusters'])}")
    if len(roots) == 0:
        print(f"{label}: tidak ada solusi feasible."); return roots
    costs = [problem.crash_cost(r) for r in roots]
    print(f"{label}: Z = {min(costs)} (gap thd ILP = {min(costs) - z_star}), "
          f"{len(roots)} solusi, {time.time() - t0:.0f}s")
    return roots


def main():
    tasks = load_tasks(DATA)
    pure = ProjectCrashingProblem(tasks, DEADLINE, params=PARAMS)
    print(f"{pure.name} | makespan normal={pure.makespan(pure.d_max)}, "
          f"min={pure.makespan(pure.d_min)}")

    z_star, _ = ilp_ground_truth(pure)
    print(f"ILP ground truth: Z* = {z_star}\n")

    roots = run_one(pure, z_star, "SOAC")
    if len(roots):
        print()
        print(pure.report(roots[:3]))


if __name__ == "__main__":
    main()
