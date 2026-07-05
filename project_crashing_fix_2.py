import json
import math
import time
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import qmc


CONFIG = {
    "activity_data_path": "activity_data_v3.json",
    "resource_requirements_path": "resource_requirements_v3.json",
    "resource_capacity_path": "resource_capacity_v3.json",
    "resource_id_map": None,
    "default_capacity_unmapped": 999, #ini kalau semisal gaada kapasitasnya, tapi mungkin better diilangin aja cuma nanti bisa error (tapi sepertinya lebih baik gini, jadi tau mana yang perlu diperbaiki sm user)
    "max_schedule_horizon": 2000,
    "rounding": "half_up",
    "cost_mode": "crash_slope",
    "m_cluster": 100,
    "k_cluster": 50,
    "r_cl": 0.95,
    "theta_cl": np.pi / 4,
    "gamma": -float("inf"),
    "num_check_points": 1,
    "sdoa_m": 50,
    "sdoa_k_max": 25,
    "sdoa_r": 0.95,
    "sdoa_theta": np.pi / 4,
    "delta": 0.05,
    "epsilon": 1e-6,
    "min_radius_safety_factor": 1.0,
}


def load_project_data(config=CONFIG):
    base = Path(config.get("data_dir", "."))
    activity_data = json.loads(Path(config["activity_data_path"]).read_text())
    resource_requirements = json.loads(Path(config["resource_requirements_path"]).read_text())
    resource_capacity = json.loads(Path(config["resource_capacity_path"]).read_text())
    task_ids = list(activity_data.keys())
    capacity_names = list(resource_capacity.keys())
    raw_ids = {rid for reqs in resource_requirements.values() for rid in reqs.keys()}
    ids_are_numeric = all(rid.isdigit() for rid in raw_ids) if raw_ids else True
    id_map = config.get("resource_id_map")
    if id_map is None:
        if ids_are_numeric:
            id_map = {str(i + 1): name for i, name in enumerate(capacity_names)}
        else:
            id_map = {rid: rid for rid in raw_ids}
    all_resource_ids = sorted(raw_ids, key=(lambda s: int(s)) if ids_are_numeric else (lambda s: s))
    unmapped_ids = [rid for rid in all_resource_ids if rid not in id_map]
    if unmapped_ids:
        warnings.warn(
            f"[project_crashing] Resource id {unmapped_ids} belum punya mapping nama resource; diberi kapasitas default {config['default_capacity_unmapped']} (tidak membatasi).",
            UserWarning,
        )
    capacity_by_id = {}
    for rid in all_resource_ids:
        name = id_map.get(rid)
        if name is not None and name in resource_capacity:
            capacity_by_id[rid] = resource_capacity[name]
        else:
            capacity_by_id[rid] = config["default_capacity_unmapped"]
    return {
        "task_ids": task_ids,
        "activity_data": activity_data,
        "resource_requirements": resource_requirements,
        "capacity_by_id": capacity_by_id,
        "id_map": id_map,
    }


def build_topological_order(task_ids, activity_data):
    indegree = {t: 0 for t in task_ids}
    successors = {t: [] for t in task_ids}
    for t in task_ids:
        for pred in activity_data[t]["required_activities"]:
            if pred not in successors:
                continue
            successors[pred].append(t)
            indegree[t] += 1
    from collections import deque
    queue = deque(t for t in task_ids if indegree[t] == 0)
    order = []
    while queue:
        t = queue.popleft()
        order.append(t)
        for s in successors[t]:
            indegree[s] -= 1
            if indegree[s] == 0:
                queue.append(s)
    if len(order) != len(task_ids):
        missing = set(task_ids) - set(order)
        raise ValueError(f"Precedence graph mengandung siklus atau task tak terjangkau: {missing}")
    return order


def round_value(x, method="half_up"):
    if method == "half_up":
        return math.floor(x + 0.5)
    if method == "half_even":
        return int(np.round(x))
    if method == "floor":
        return math.floor(x)
    raise ValueError(f"Metode pembulatan tidak dikenal: {method}")


def decode_duration(x_j, min_time, normal_time, rounding="half_up"):
    expanded_lo = min_time - 0.5
    expanded_hi = normal_time + 0.5
    scaled = expanded_lo + x_j * (expanded_hi - expanded_lo)
    d = round_value(scaled, rounding)
    return int(np.clip(d, min_time, normal_time))


def find_feasible_start(earliest_start, duration, resource_reqs, capacity_by_id,
                         usage, max_horizon):
    if not resource_reqs:
        return earliest_start
    t = earliest_start
    while t <= max_horizon:
        conflict_day = None
        for day in range(t, t + duration):
            for rid, qty in resource_reqs.items():
                cap = capacity_by_id.get(rid, 0)
                used = usage.get((rid, day), 0)
                if used + qty > cap:
                    conflict_day = day
                    break
            if conflict_day is not None:
                break
        if conflict_day is None:
            return t
        t = conflict_day + 1
    return earliest_start


def evaluate_schedule(x, project):
    task_ids = project["task_ids"]
    activity_data = project["activity_data"]
    resource_requirements = project["resource_requirements"]
    capacity_by_id = project["capacity_by_id"]
    topo_order = project["topo_order"]
    rounding = project["rounding"]
    max_horizon = project["max_schedule_horizon"]
    idx = project["task_index"]
    cost_mode = project.get("cost_mode", "crash_slope")

    x = np.clip(x, 0.0, 1.0)

    start, end, duration = {}, {}, {}
    usage = {}
    total_cost = 0.0
    feasible = True

    for t in topo_order:
        act = activity_data[t]
        min_time, normal_time = act["activity_min_time"], act["activity_normal_time"]
        x_j = x[idx[t]]
        d_j = decode_duration(x_j, min_time, normal_time, rounding)

        preds = act["required_activities"]
        earliest_start = max((end[p] for p in preds if p in end), default=0)

        reqs = resource_requirements.get(t, {})
        s_j = find_feasible_start(earliest_start, d_j, reqs, capacity_by_id, usage, max_horizon)
        if s_j + d_j > max_horizon:
            feasible = False

        e_j = s_j + d_j
        start[t], end[t], duration[t] = s_j, e_j, d_j

        for rid, qty in reqs.items():
            for day in range(s_j, e_j):
                usage[(rid, day)] = usage.get((rid, day), 0) + qty

        if cost_mode == "crash_slope":
            total_cost += act["crash_cost"] * (normal_time - d_j)
        elif cost_mode == "direct":
            total_cost += act["crash_cost"] * d_j
        else:
            raise ValueError(f"cost_mode tidak dikenal: {cost_mode}")

    return {
        "start": start, "end": end, "duration": duration,
        "total_cost": total_cost, "feasible": feasible,
        "makespan": max(end.values()) if end else 0,
    }


class ProjectCrashingProblem:
    problem_type = "PROJECT_CRASHING"

    def __init__(self, project, penalty=1e12, target_duration=None, penalty_per_day=1e7):
        self.project = project
        self.n_var = len(project["task_ids"])
        self.domain = [(0.0, 1.0)] * self.n_var
        self.penalty = penalty
        self.target_duration = target_duration
        self.penalty_per_day = penalty_per_day
        self.n_evals = 0

    def evaluate_fitness(self, x):
        self.n_evals += 1
        x = np.asarray(x, dtype=float)
        out_of_box = np.any((x < -1e-9) | (x > 1 + 1e-9))
        result = evaluate_schedule(x, self.project)
        cost = result["total_cost"]
        if not result["feasible"]:
            cost += self.penalty
        if self.target_duration is not None:
            overrun_days = max(0, result["makespan"] - self.target_duration)
            cost += self.penalty_per_day * overrun_days
        if out_of_box:
            overshoot = np.sum(np.clip(-x, 0, None) + np.clip(x - 1, 0, None))
            cost += self.penalty * overshoot
        return -cost

    def evaluate_fitness_batch(self, points):
        return np.array([self.evaluate_fitness(p) for p in points])


def build_project(config=CONFIG):
    data = load_project_data(config)
    data["topo_order"] = build_topological_order(data["task_ids"], data["activity_data"])
    data["task_index"] = {t: i for i, t in enumerate(data["task_ids"])}
    data["rounding"] = config["rounding"]
    data["max_schedule_horizon"] = config["max_schedule_horizon"]
    data["cost_mode"] = config.get("cost_mode", "crash_slope")
    return data


def generate_sobol_points(num_points, dimension, domain, augment_boundary=False):
    lower_bounds = np.array([d[0] for d in domain])
    upper_bounds = np.array([d[1] for d in domain])
    try:
        sampler = qmc.Sobol(d=dimension, scramble=False)
        unit_points = sampler.random(n=num_points)
        points = qmc.scale(unit_points, lower_bounds, upper_bounds)
    except ValueError:
        warnings.warn("Sobol sequence generation failed. Falling back to uniform.", UserWarning)
        points = np.random.uniform(lower_bounds, upper_bounds, (num_points, dimension))
    if augment_boundary and dimension <= 12:
        corners = np.array(np.meshgrid(*[(lo, hi) for lo, hi in domain])).T.reshape(-1, dimension)
        points = np.vstack([corners, points])
    return points


def get_rotation_matrix(n, theta):
    if n == 1:
        return np.identity(1)
    R_total = np.identity(n)
    c, s = np.cos(theta), np.sin(theta)
    for i in range(n - 2, -1, -1):
        for j in range(i, -1, -1):
            p = n - i - 2
            q = n - j - 1
            R_pq = np.identity(n)
            R_pq[p, p] = c
            R_pq[p, q] = -s
            R_pq[q, p] = s
            R_pq[q, q] = c
            R_total = R_pq @ R_total
    return R_total


def is_in_domain(point, domain):
    for i, (lo, hi) in enumerate(domain):
        if not (lo <= point[i] <= hi):
            return False
    return True


def filter_unique_roots(candidates, delta):
    if not candidates:
        return np.array([])
    sorted_candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
    final_roots = []
    for cand, f_val in sorted_candidates:
        found_close = False
        for i, (existing, existing_f) in enumerate(final_roots):
            if np.linalg.norm(cand - existing) <= delta:
                found_close = True
                if f_val > existing_f:
                    final_roots[i] = (cand, f_val)
                break
        if not found_close:
            final_roots.append((cand, f_val))
    return np.array([root for root, _ in final_roots])


def compute_min_cluster_radius(project, safety_factor=1.0):
    widths = []
    for t, act in project["activity_data"].items():
        w = (act["activity_normal_time"] - act["activity_min_time"]) + 1
        widths.append(w)
    max_width = max(widths) if widths else 1
    return safety_factor * (1.0 / max_width)


class Cluster:
    def __init__(self, center, radius, min_radius=0.0):
        self.center = np.array(center, dtype=float)
        self.min_radius = float(min_radius)
        self.radius = max(float(radius), self.min_radius)

    def __repr__(self):
        return f"Cluster(center[:5]={self.center[:5].round(4)}..., radius={self.radius:.5f})"


def spiral_dynamics_optimization(objective_func, domain, params, minimization=False,
                                  custom_initial_points=None):
    m = params.get('m', 20)
    r = params.get('r', 0.95)
    theta = params.get('theta', np.pi / 4)
    k_max = params.get('k_max', 100)
    n = len(domain)

    if custom_initial_points is not None:
        search_points = np.array(custom_initial_points)
        m = len(search_points)
    else:
        search_points = generate_sobol_points(m, n, domain)

    R_n = get_rotation_matrix(n, theta)
    S_n = r * R_n
    I_n = np.identity(n)

    def evaluate(points):
        return np.array([objective_func(p) for p in points])

    best_values = evaluate(search_points)
    best_idx = np.argmin(best_values) if minimization else np.argmax(best_values)
    x_star = search_points[best_idx].copy()
    best_value = best_values[best_idx]

    for k in range(k_max):
        search_points = search_points @ S_n.T - (S_n - I_n) @ x_star
        current_values = evaluate(search_points)
        current_best_idx = (np.argmin(current_values) if minimization
                             else np.argmax(current_values))
        current_best_value = current_values[current_best_idx]
        if (minimization and current_best_value < best_value) or \
           (not minimization and current_best_value > best_value):
            x_star = search_points[current_best_idx].copy()
            best_value = current_best_value

    return x_star


def process_point_for_clustering(y, clusters, problem, gamma, params, min_radius,
                                  history=None):
    F_y = problem.evaluate_fitness(y)
    if F_y <= gamma:
        return clusters
    if not clusters:
        initial_radius = 0.5 * min(hi - lo for lo, hi in problem.domain)
        clusters.append(Cluster(y, initial_radius, min_radius))
        return clusters
    centers = np.array([c.center for c in clusters])
    dists = np.linalg.norm(centers - y, axis=1)
    closest_idx = np.argmin(dists)
    nearest_cluster = clusters[closest_idx]
    x_C = nearest_cluster.center
    F_xC = problem.evaluate_fitness(x_C)
    num_check_points = params.get('num_check_points', 1)
    t_vals = [i / (num_check_points + 1) for i in range(1, num_check_points + 1)]
    x_ts = [y + t * (x_C - y) for t in t_vals]
    F_xts = [problem.evaluate_fitness(xt) for xt in x_ts]
    F_xt_min = min(F_xts)
    F_xt_max = max(F_xts)
    dist_half = np.linalg.norm(y - x_C) / 2.0
    if F_xt_min < F_y and F_xt_min < F_xC:
        clusters.append(Cluster(y.copy(), dist_half, min_radius))
    elif F_xt_max > F_y and F_xt_max > F_xC:
        clusters.append(Cluster(y.copy(), dist_half, min_radius))
        x_t_best = x_ts[int(np.argmax(F_xts))]
        clusters = process_point_for_clustering(x_t_best, clusters, problem, gamma, params, min_radius, history)
    elif F_y > F_xC:
        nearest_cluster.center = y.copy()
    nearest_cluster.radius = max(dist_half, min_radius)
    return clusters


def perform_iterative_clustering(problem, params, min_radius):
    m_cluster = params['m_cluster']
    gamma_cfg = params.get('gamma', -float('inf'))
    k_cluster = params['k_cluster']
    r = params.get('r_cl', 0.95)
    theta = params.get('theta_cl', np.pi / 4)
    n = problem.n_var
    domain = problem.domain
    points = generate_sobol_points(m_cluster, n, domain)
    m_total = len(points)
    R_n = get_rotation_matrix(n, theta)
    S_n = r * R_n
    I_n = np.identity(n)
    clusters = []
    F_values = problem.evaluate_fitness_batch(points)
    best_idx = np.argmax(F_values)
    x_prime = points[best_idx].copy()
    initial_radius = 0.5 * min(hi - lo for lo, hi in domain)
    clusters.append(Cluster(x_prime, initial_radius, min_radius))
    for k in range(k_cluster):
        F_values = problem.evaluate_fitness_batch(points)
        F_best = np.max(F_values)
        if gamma_cfg is not None and gamma_cfg != -float('inf'):
            cutoff = gamma_cfg * F_best if F_best > 0 else gamma_cfg
        else:
            cutoff = -float('inf')
        for i in range(m_total):
            if not is_in_domain(points[i], domain):
                continue
            if F_values[i] > cutoff:
                centers = np.array([c.center for c in clusters])
                is_center = np.any(np.all(np.abs(centers - points[i]) < 1e-8, axis=1))
                if not is_center:
                    clusters = process_point_for_clustering(points[i], clusters, problem, cutoff, params, min_radius)
        F_values = problem.evaluate_fitness_batch(points)
        x_p = points[np.argmax(F_values)].copy()
        points = points @ S_n.T - (S_n - I_n) @ x_p
    return clusters


def run_sdoa_on_clusters(clusters, problem, params):
    sdoa_params = {
        'm': params.get('sdoa_m', 40),
        'r': params.get('sdoa_r', 0.95),
        'theta': params.get('sdoa_theta', np.pi / 4),
        'k_max': params.get('sdoa_k_max', 80),
    }
    n = problem.n_var
    domain = problem.domain
    candidates = []
    for cluster in clusters:
        cluster_domain = []
        for dim in range(n):
            lo = max(domain[dim][0], cluster.center[dim] - cluster.radius)
            hi = min(domain[dim][1], cluster.center[dim] + cluster.radius)
            cluster_domain.append((lo, hi))
        if any(hi - lo < 1e-12 for lo, hi in cluster_domain):
            candidates.append(cluster.center.copy())
            continue
        initial_points = generate_sobol_points(sdoa_params['m'], n, cluster_domain)
        candidate = spiral_dynamics_optimization(
            objective_func=problem.evaluate_fitness,
            domain=cluster_domain,
            params=sdoa_params,
            minimization=False,
            custom_initial_points=initial_points,
        )
        candidates.append(candidate)
    return np.array(candidates)


def select_final_optimal(candidates, problem, params):
    delta = params.get('delta', 0.05)
    epsilon = params.get('epsilon', 1e-6)
    gamma = params.get('gamma', None)
    domain = problem.domain
    if candidates is None or len(candidates) == 0:
        return np.array([])
    evals = [problem.evaluate_fitness(c) for c in candidates if is_in_domain(c, domain)]
    F_star = max(evals) if evals else 0
    accurate_candidates = []
    for cand in candidates:
        if not is_in_domain(cand, domain):
            continue
        f_val = problem.evaluate_fitness(cand)
        if gamma is not None and gamma != -float('inf') and F_star > 0:
            if f_val <= (1.0 - epsilon) * F_star:
                continue
        accurate_candidates.append((cand, f_val))
    return filter_unique_roots(accurate_candidates, delta)


def collect_optimal_solutions(results, target_duration, cost_tol=1e-6):
    """
    Dari daftar `results` (hasil solve_single_problem, sudah berisi semua
    kandidat unik hasil clustering+SDOA), ambil SEMUA solusi yang:
      - benar-benar feasible DAN makespan aktualnya <= target_duration
        (bukan cuma lolos threshold penalti di fitness -- di-cek ulang di sini
        supaya solusi yang "kebetulan lolos" gara-gara penalti kurang besar
        tidak ikut ke dalam daftar solusi optimal),
      - punya total_cost SAMA (dalam toleransi cost_tol) dengan cost minimum
        yang ditemukan untuk target_duration ini -- inilah "kombinasi
        optimal jamak" yang dicari (alasan pakai multimodal search),
      - UNIK dari sisi kombinasi durasi tiap task (dedup pakai tuple durasi
        per task). Ini perlu karena beberapa titik x berbeda hasil SDOA bisa
        saja ter-decode ke kombinasi durasi integer yang identik.

    Return
    ------
    list of dict (format sama seperti 1 elemen `results`), semuanya punya
    cost minimum yang sama, sudah di-dedup dan diurutkan biar konsisten
    (urutan berdasarkan urutan task_ids lalu durasi, supaya output stabil
    antar run).
    """
    valid = [r for r in results if r["feasible"] and r["makespan"] <= target_duration]
    if not valid:
        return []

    min_cost = min(r["total_cost"] for r in valid)
    tied = [r for r in valid if abs(r["total_cost"] - min_cost) <= cost_tol]

    seen = set()
    unique = []
    for r in tied:
        key = tuple(sorted(r["duration"].items()))
        if key not in seen:
            seen.add(key)
            unique.append(r)

    unique.sort(key=lambda r: tuple(sorted(r["duration"].items())))
    return unique

def enforce_monotonic_tradeoff(curve):
    """
    Cost minimum crashing bersifat MONOTON TIDAK NAIK terhadap target_duration:
    kalau D dilonggarkan, himpunan solusi feasible adalah superset dari D yang
    lebih ketat, jadi cost minimum yang bisa dicapai tidak pernah boleh naik.

    Karena tiap D di-solve independen & stokastik (Sobol + spiral dengan
    budget terbatas), kadang search GAGAL menemukan solusi sebaik yang
    ditemukan untuk D yang lebih kecil -> hasil sweep bisa non-monoton
    (cost "naik" di suatu titik). Itu bukan sifat asli persoalannya, itu
    kegagalan pencarian. Fungsi ini memperbaikinya dengan MEWARISKAN solusi
    dari D lebih kecil (yang otomatis tetap valid untuk D ini) setiap kali
    hasil independen untuk D saat ini ternyata tidak lebih baik.
    """
    curve_sorted = sorted(curve, key=lambda r: r["target_duration"])
    best_cost = None
    best_solutions = None
    best_makespan = None

    fixed = []
    for row in curve_sorted:
        new_row = dict(row)
        if not row["feasible"]:
            fixed.append(new_row)
            continue

        if best_cost is None or row["best_cost"] < best_cost - 1e-9:
            best_cost = row["best_cost"]
            best_solutions = row["solutions"]
            best_makespan = row["achieved_makespan"]
            new_row["inherited_from_smaller_D"] = False
        else:
            new_row["best_cost"] = best_cost
            new_row["achieved_makespan"] = best_makespan
            new_row["solutions"] = best_solutions
            new_row["n_optimal_solutions"] = len(best_solutions)
            new_row["inherited_from_smaller_D"] = True

        fixed.append(new_row)

    return fixed

def solve_single_problem(problem, config, verbose=True):
    t0 = time.time()
    project = problem.project
    min_radius = compute_min_cluster_radius(project, config.get('min_radius_safety_factor', 1.0))
    clusters = perform_iterative_clustering(problem, config, min_radius)
    if verbose:
        print(f"  clustering: {len(clusters)} cluster ({problem.n_evals} evaluasi objective sejauh ini)")
    candidates = run_sdoa_on_clusters(clusters, problem, config)
    if verbose:
        print(f"  SDOA lokal: {len(candidates)} kandidat ({problem.n_evals} evaluasi objective total)")
    roots = select_final_optimal(candidates, problem, config)
    if verbose:
        print(f"  seleksi akhir: {len(roots)} solusi unik")
    results = []
    for x in roots:
        sched = evaluate_schedule(np.clip(x, 0, 1), project)
        results.append({
            "x": x, "total_cost": sched["total_cost"], "makespan": sched["makespan"],
            "feasible": sched["feasible"], "start": sched["start"], "end": sched["end"],
            "duration": sched["duration"],
        })
    results.sort(key=lambda r: r["total_cost"])
    elapsed = time.time() - t0
    if verbose:
        print(f"  selesai dalam {elapsed:.2f}s")
    return results


def solve_time_cost_tradeoff(config=CONFIG, target_durations=None, penalty_per_day=1e7,
                              sweep_config_overrides=None, cost_tol=1e-6, verbose=True):
    project = build_project(config)
    n = len(project["task_ids"])
    baseline_hi = evaluate_schedule(np.ones(n), project)
    baseline_lo = evaluate_schedule(np.zeros(n), project)
    M_normal, M_min = baseline_hi["makespan"], baseline_lo["makespan"]
    if target_durations is None:
        target_durations = list(range(M_min, M_normal + 1))
    sweep_config = dict(config)
    if sweep_config_overrides:
        sweep_config.update(sweep_config_overrides)
    if verbose:
        print(f"[tradeoff] makespan normal={M_normal} hari, makespan minimum (full crash)={M_min} hari")
        print(f"[tradeoff] sweep {len(target_durations)} target durasi...")
    curve = []
    for D in target_durations:
        problem = ProjectCrashingProblem(project, target_duration=D, penalty_per_day=penalty_per_day)
        results = solve_single_problem(problem, sweep_config, verbose=False)
        optimal_solutions = collect_optimal_solutions(results, D, cost_tol=cost_tol)
        crash_days = M_normal - D

        if optimal_solutions:
            best_cost = optimal_solutions[0]["total_cost"]
            achieved_makespan = optimal_solutions[0]["makespan"]
            feasible = True
        else:
            best_cost = None
            achieved_makespan = None
            feasible = False

        row = {
            "target_duration": D,
            "crash_days": crash_days,
            "best_cost": best_cost,
            "achieved_makespan": achieved_makespan,
            "feasible": feasible,
            "n_optimal_solutions": len(optimal_solutions),
            "solutions": optimal_solutions,
            # dipertahankan untuk kompatibilitas kode lama yang mengakses row["schedule"]
            "schedule": optimal_solutions[0] if optimal_solutions else None,
        }
        curve.append(row)
        if verbose:
            status = "OK" if feasible else "TIDAK TERCAPAI"
            cost_str = f"{best_cost:.2f}" if best_cost is not None else "N/A"
            print(f"  D={D:>4} (crash {crash_days:>3} hari dari normal) -> "
                  f"cost={cost_str}  makespan tercapai={achieved_makespan}  "
                  f"[{status}]  ({row['n_optimal_solutions']} kombinasi optimal ditemukan)")
    curve = enforce_monotonic_tradeoff(curve)
    if verbose:
        n_inherited = sum(1 for r in curve if r.get("inherited_from_smaller_D"))
        if n_inherited:
            print(f"[tradeoff] {n_inherited} dari {len(curve)} titik D diperbaiki "
                  f"(cost diwariskan dari D lebih kecil karena search independen gagal "
                  f"menemukan yang sebaik itu)")
    return {"project": project, "curve": curve, "M_normal": M_normal, "M_min": M_min}


def format_one_combo_table(sched, project):
    """Format 1 kombinasi (1 elemen dari row['solutions']) jadi tabel per-task."""
    activity_data = project["activity_data"]
    lines = []
    lines.append(f"    {'Task':<14}{'Normal':>7}{'Aktual':>7}{'Crash(hr)':>10}"
                 f"{'Biaya Crash':>14}{'Mulai':>8}{'Selesai':>9}")
    lines.append("    " + "-" * 70)

    task_rows = []
    for t in project["task_ids"]:
        act = activity_data[t]
        normal_t = act["activity_normal_time"]
        actual_d = sched["duration"][t]
        crashed_days = normal_t - actual_d
        task_cost = crashed_days * act["crash_cost"]
        s, e = sched["start"][t], sched["end"][t]
        task_rows.append((t, normal_t, actual_d, crashed_days, task_cost, s, e))

    task_rows_by_start = sorted(task_rows, key=lambda r: (r[5], r[0]))
    for (t, normal_t, actual_d, crashed_days, task_cost, s, e) in task_rows_by_start:
        crash_marker = f"{crashed_days}" if crashed_days > 0 else "-"
        cost_marker = f"{task_cost:,.0f}" if crashed_days > 0 else "-"
        lines.append(f"    {t:<14}{normal_t:>7}{actual_d:>7}{crash_marker:>10}"
                     f"{cost_marker:>14}{s:>8}{e:>9}")

    order_str = " -> ".join(f"{t}(h{s})" for (t, _, _, _, _, s, _) in task_rows_by_start)
    lines.append(f"\n    Urutan pengerjaan: {order_str}")

    crashed_only = [r for r in task_rows if r[3] > 0]
    if crashed_only:
        crash_list = ", ".join(f"{t} (-{c}hr, +Rp{cost:,.0f})"
                                for (t, _, _, c, cost, _, _) in crashed_only)
        lines.append(f"    Task yang di-crash: {crash_list}")
    else:
        lines.append("    Task yang di-crash: (tidak ada, semua durasi normal)")

    return "\n".join(lines)


def format_all_combos_for_duration(row, project):
    """
    Format 1 baris kurva time-cost tradeoff, MENAMPILKAN SEMUA kombinasi
    yang sama-sama optimal (tied minimum cost) untuk target_duration ini --
    bukan cuma kombinasi pertama yang ditemukan solver.
    """
    lines = []
    n_sol = row["n_optimal_solutions"]

    if not row["feasible"] or n_sol == 0:
        lines.append(f"=== Target durasi = {row['target_duration']} hari "
                      f"-- TIDAK ADA SOLUSI FEASIBLE ditemukan ===")
        return "\n".join(lines)

    inherited_note = (" [solusi diwariskan dari D lebih kecil -- pencarian independen "
                       "untuk D ini tidak menemukan yang sebaik itu]"
                       if row.get("inherited_from_smaller_D") else "")
    header = (f"=== Target durasi = {row['target_duration']} hari "
              f"(crash {row['crash_days']} hari dari normal) "
              f"| Biaya crash optimal = Rp {row['best_cost']:,.0f} "
              f"| Makespan tercapai = {row['achieved_makespan']} hari "
              f"| {n_sol} kombinasi sama-sama optimal ditemukan{inherited_note} ===")
    lines.append(header)

    for i, sched in enumerate(row["solutions"], start=1):
        lines.append(f"\n  --- Kombinasi {i} dari {n_sol} ---")
        lines.append(format_one_combo_table(sched, project))

    return "\n".join(lines)


def build_full_tradeoff_report(out):
    """Bangun laporan lengkap (string) untuk seluruh kurva -- dipakai untuk print & simpan ke .txt."""
    lines = []
    lines.append("=" * 78)
    lines.append(" LAPORAN RINCI TIME-COST TRADEOFF PER TARGET DURASI")
    lines.append(f" Makespan normal = {out['M_normal']} hari | "
                 f"Makespan minimum (full crash) = {out['M_min']} hari")
    lines.append("=" * 78)
    for row in out["curve"]:
        lines.append("")
        lines.append(format_all_combos_for_duration(row, out["project"]))
    return "\n".join(lines)


def print_full_tradeoff_report(out):
    print(build_full_tradeoff_report(out))


def save_tradeoff_report_to_txt(out, filepath="laporan_time_cost_tradeoff.txt"):
    """
    Simpan laporan lengkap ke file .txt (dipakai kalau outputnya terlalu
    panjang untuk terminal, misalnya untuk proyek besar 126-task).
    """
    report_str = build_full_tradeoff_report(out)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report_str)
    abs_path = str(Path(filepath).resolve())
    print(f"\n[info] Laporan lengkap disimpan ke: {abs_path}")
    return abs_path


if __name__ == "__main__":
    project_tmp = build_project(CONFIG)
    n_tmp = len(project_tmp["task_ids"])
    hi = evaluate_schedule(np.ones(n_tmp), project_tmp)
    lo = evaluate_schedule(np.zeros(n_tmp), project_tmp)
    print("makespan normal (x=1):", hi["makespan"], "cost:", hi["total_cost"])
    print("makespan full-crash (x=0):", lo["makespan"], "cost:", lo["total_cost"])
    sample_durations = [240, 241, 242, 243, 244]#list(range(lo["makespan"], hi["makespan"] + 1))
    print("sample_durations:", sample_durations)
    out = solve_time_cost_tradeoff(
        CONFIG,
        target_durations=sample_durations,
        verbose=True,
    )

    print("\n=== RINGKASAN ===")
    for row in out["curve"]:
        print(row["target_duration"], row["crash_days"], row["best_cost"], row["achieved_makespan"])

    print_full_tradeoff_report(out)
    save_tradeoff_report_to_txt(out, filepath="laporan_time_cost_tradeoff.txt")
