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
import scipy.sparse as sp
from scipy.optimize import milp, LinearConstraint, Bounds

from pysne.solver import solve_system
from project_crashing_problem_fix import ProjectCrashingProblem
# from project_crashing_problem_fix_try_192 import ProjectCrashingProblem # uncomment this if you want to use the new version (faster)

# DATA = "E:/p2ms/IMW26-AAA/adiel/data/activity_data_v3.json"
current_dir = Path(__file__).resolve().parent
DATA = current_dir / "activity_data_v3.json"
CAP_DATA = current_dir / "resource_capacity_v3.json"
REQ_DATA = current_dir / "resource_requirements_v3.json"

DEADLINE = 246  # makespan normal = 249 -> perlu crash jalur kritis 6 hari


def load_tasks(path):
    raw = json.load(open(path))
    return [{"name": k,
             "predecessors": v["required_activities"],
             "d_min": v["activity_min_time"],
             "d_max": v["activity_normal_time"],
             "crash_cost": v["crash_cost"]} for k, v in raw.items()]


def ilp_ground_truth(problem):
    """Kembalikan (Z_optimal, d_optimal) via MILP.

    - Tanpa sumber daya: LP start-kontinu ringan (persis versi lama).
    - Dengan sumber daya: MILP time-indexed multi-mode + kendala cumulative.
    """
    n = problem.n_tasks
    T = problem.deadline
    dmin, dmax, c, pred = problem.d_min, problem.d_max, problem.c, problem.pred_idx

    if not problem.has_resources:
        c_obj = np.concatenate([-c, np.zeros(n)])
        A, lb, ub = [], [], []
        for j, preds in enumerate(pred):
            for p in preds:
                row = np.zeros(2 * n)
                row[n + j], row[n + p], row[p] = 1, -1, -1
                A.append(row); lb.append(0); ub.append(np.inf)
        for j in range(n):
            row = np.zeros(2 * n)
            row[n + j], row[j] = 1, 1
            A.append(row); lb.append(-np.inf); ub.append(T)
        cons = LinearConstraint(np.array(A), lb, ub)
        bounds = Bounds(np.concatenate([dmin, np.zeros(n)]),
                        np.concatenate([dmax, np.full(n, np.inf)]))
        integ = np.concatenate([np.ones(n), np.zeros(n)])
        res = milp(c=c_obj, constraints=cons, bounds=bounds, integrality=integ)
        d = np.rint(res.x[:n]).astype(int)
        return problem.crash_cost(d), d

    # ---- RCPSP-TCT: time-indexed multi-mode ----
    Req, Cap, R = problem.Req, problem.Cap, len(problem.Cap)
    topo, succ = problem.topo_order, problem.succ_idx
    ES = np.zeros(n, int)
    for j in topo:
        ES[j] = max((ES[p] + dmin[p] for p in pred[j]), default=0)
    LF = np.full(n, T, int)
    for j in reversed(topo):
        ss = [LF[s] - dmin[s] for s in succ[j]]
        LF[j] = min(ss) if ss else T
    LS = np.minimum(LF - dmin, T - dmin)

    cols = [(j, t, dd)
            for j in range(n)
            for t in range(int(ES[j]), int(LS[j]) + 1)
            for dd in range(int(dmin[j]), int(dmax[j]) + 1)
            if t + dd <= T]
    V = len(cols)
    obj = np.array([c[j] * (dmax[j] - dd) for (j, t, dd) in cols], float)

    data, ri, ci, ld, ud, r = [], [], [], [], [], 0
    for j in range(n):                              # tiap task mulai tepat sekali
        for k, (jj, t, dd) in enumerate(cols):
            if jj == j:
                data.append(1); ri.append(r); ci.append(k)
        ld.append(1); ud.append(1); r += 1
    for j in range(n):                              # precedence: start_j >= start_p + dur_p
        for p in pred[j]:
            for k, (jj, t, dd) in enumerate(cols):
                if jj == j:
                    data.append(t); ri.append(r); ci.append(k)
                elif jj == p:
                    data.append(-(t + dd)); ri.append(r); ci.append(k)
            ld.append(0); ud.append(np.inf); r += 1
    for ridx in range(R):                           # kapasitas cumulative
        js = [j for j in range(n) if Req[j, ridx] > 0]
        if not js:
            continue
        for tau in range(T):
            any_ = False
            for k, (jj, t, dd) in enumerate(cols):
                if jj in js and t <= tau < t + dd:
                    data.append(int(Req[jj, ridx])); ri.append(r); ci.append(k); any_ = True
            if any_:
                ld.append(-np.inf); ud.append(int(Cap[ridx])); r += 1
    A = sp.csr_matrix((data, (ri, ci)), shape=(r, V))
    cons = LinearConstraint(A, np.array(ld), np.array(ud))
    res = milp(c=obj, constraints=cons, bounds=Bounds(np.zeros(V), np.ones(V)),
               integrality=np.ones(V), options={"time_limit": 300})
    xr = np.rint(res.x).astype(int)
    d = np.zeros(n, int)
    for k, (j, t, dd) in enumerate(cols):
        if xr[k] == 1:
            d[j] = dd
    return problem.crash_cost(d), d


# Parameter SOAC. Untuk varian RCPSP tiap evaluasi fitness memanggil SGS
# (lebih mahal dari CPM murni), jadi setelan cluster/iterasi dibuat lebih hemat.
# Perbesar bila ingin cakupan plateau lebih lengkap (dengan biaya waktu).
PARAMS = dict(
    m_cluster=2048, k_cluster=500, gamma=0.85,
    r_cl=0.975, theta_cl=np.pi/4,
    sdoa_m=2048, sdoa_k_max=400,
    sdoa_r=0.95, sdoa_theta=np.pi/8,
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


def build_solution_json(prob, d, solution_label="Solusi #1"):
    """Konversi vektor durasi ke dict terstruktur per task."""
    d = np.asarray(d, dtype=int)
    s, e = prob.schedule(d)
    y = prob.d_max - d
    tasks_detail = []
    for j in range(prob.n_tasks):
        tasks_detail.append({
            "task_name": prob.task_names[j],
            "duration": int(d[j]),
            "normal_duration": int(prob.d_max[j]),
            "min_duration": int(prob.d_min[j]),
            "crash_days": int(y[j]),
            "crash_cost_per_day": float(prob.c[j]),
            "crash_cost_total": float(prob.c[j] * y[j]),
            "start": int(s[j]),
            "end": int(e[j]),
        })
    return {
        "label": solution_label,
        "total_crash_cost": float(prob.crash_cost(d)),
        "makespan": int(e.max()),
        "deadline": prob.deadline,
        "resource_feasible": prob.has_resources,
        "tasks": tasks_detail,
    }


def export_report_json(prob, d_ilp, roots, z_star, ilp_time, spoc_time):
    """Simpan laporan ILP vs SPOC side-by-side ke JSON di folder sweep."""
    output_dir = current_dir / "outputs_azarya" / "sweep"
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- ILP solution ---
    ilp_solution = build_solution_json(prob, d_ilp, "ILP Ground Truth")
    ilp_solution["solver"] = "ILP (scipy.optimize.milp)"
    ilp_solution["elapsed_seconds"] = round(ilp_time, 2)

    # --- SPOC solutions ---
    spoc_solutions = []
    for k, root in enumerate(roots):
        sol = build_solution_json(prob, root, f"SPOC Solusi #{k + 1}")
        sol["solver"] = "SPOC (SOAC)"
        spoc_solutions.append(sol)

    # --- Perbandingan per task ---
    d_ilp_arr = np.asarray(d_ilp, dtype=int)
    best_spoc = np.asarray(roots[0], dtype=int) if len(roots) else None
    comparison = {
        "ilp_cost": z_star,
        "spoc_best_cost": float(prob.crash_cost(best_spoc)) if best_spoc is not None else None,
        "gap": float(prob.crash_cost(best_spoc) - z_star) if best_spoc is not None else None,
        "duration_identical": bool(np.array_equal(d_ilp_arr, best_spoc)) if best_spoc is not None else False,
        "num_spoc_solutions": len(roots),
    }
    if best_spoc is not None:
        task_comparison = []
        for j in range(prob.n_tasks):
            task_comparison.append({
                "task_name": prob.task_names[j],
                "ilp_duration": int(d_ilp_arr[j]),
                "spoc_duration": int(best_spoc[j]),
                "match": bool(d_ilp_arr[j] == best_spoc[j]),
            })
        comparison["tasks"] = task_comparison

    report = {
        "deadline": prob.deadline,
        "n_tasks": prob.n_tasks,
        "has_resources": prob.has_resources,
        "ilp_solution": ilp_solution,
        "spoc_solutions": spoc_solutions,
        "comparison": comparison,
    }

    filename = f"report_ilp_vs_spoc_T{prob.deadline}.json"
    filepath = output_dir / filename
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[JSON] Laporan disimpan: {filepath}")
    return filepath

def main():
    tasks = load_tasks(DATA)
    resource_capacity = json.load(open(CAP_DATA))
    resource_requirements = json.load(open(REQ_DATA))

    prob = ProjectCrashingProblem(
        tasks, DEADLINE, params=PARAMS,
        resource_capacity=resource_capacity,
        resource_requirements=resource_requirements,
    )
    print(f"{prob.name} | makespan normal (CPM)={prob.makespan(prob.d_max)}, "
          f"min (CPM)={prob.makespan(prob.d_min)}")
    print(f"Sumber daya aktif: {len(prob.resource_names)} jenis "
          f"(kapasitas ditegakkan via SGS pada makespan).")
    t0 = time.time()
    z_star, d_ilp = ilp_ground_truth(prob)
    ilp_time = time.time() - t0
    print(f"ILP ground truth (RCPSP-TCT): Z* = {z_star}  "
          f"[{ilp_time:.1f}s]")
    print()
    print("=" * 70)
    print("  LAPORAN SOLUSI ILP (Ground Truth)")
    print("=" * 70)
    print(prob.report([d_ilp]))
    print("=" * 70)
    print()

    t1 = time.time()
    roots = run_one(prob, z_star, "SOAC")
    spoc_time = time.time() - t1
    if len(roots):
        print()
        print("=" * 70)
        print("  LAPORAN SOLUSI SPOC")
        print("=" * 70)
        print(prob.report(roots))  # Removed [:3] to show all solutions
        print("=" * 70)

        # --- Perbandingan ILP vs SPOC ---
        d_ilp_arr = np.asarray(d_ilp, dtype=int)
        best_spoc = np.asarray(roots[0], dtype=int)
        
        # Check if ANY of the SPOC solutions match the ILP exactly
        match_idx = -1
        for i, r in enumerate(roots):
            if np.array_equal(d_ilp_arr, np.asarray(r, dtype=int)):
                match_idx = i
                break
                
        z_spoc = prob.crash_cost(best_spoc)
        print()
        print("=" * 70)
        print("  PERBANDINGAN ILP vs SPOC")
        print("=" * 70)
        print(f"  Z* ILP   = {z_star:.2f}")
        print(f"  Z  SPOC  = {z_spoc:.2f}")
        print(f"  Gap      = {z_spoc - z_star:.2f}")
        
        if match_idx != -1:
            print(f"  Durasi identik? YA (Match dengan SPOC Solusi #{match_idx + 1})")
        else:
            print(f"  Durasi identik? TIDAK (Tidak ada dari {len(roots)} solusi SPOC yang identik dengan ILP)")
            diffs = np.where(d_ilp_arr != best_spoc)[0]
            print(f"  Task berbeda (dibandingkan dengan SPOC Solusi #1) ({len(diffs)}):")
            for j in diffs:
                print(f"    {prob.task_names[j]:<22} "
                      f"ILP d={d_ilp_arr[j]:>3}  vs  SPOC d={best_spoc[j]:>3}")
        print("=" * 70)

        # --- Export JSON report ---
        export_report_json(prob, d_ilp, roots, z_star, ilp_time, spoc_time)

if __name__ == "__main__":
    main()
