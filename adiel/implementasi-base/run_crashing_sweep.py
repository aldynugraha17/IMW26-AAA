#!/usr/bin/env python3
"""
run_crashing_sweep.py
=====================
For each feasible project deadline T (from normal makespan down to min makespan):
  1. Solve cost_with_deadline at T  → minimum crash cost C(T)
  2. Find all Diophantine solutions that sum to C(T)
  3. Evaluate min-makespan for every solution (CP-SAT)
  4. Save the makespan distribution plot immediately (T_<T>_dist.png)
  5. For solutions whose makespan == T, save a Gantt chart (T_<T>_gantt_sol<i>.png)

Outputs are written to:
  adiel/outputs/sweep/
    T_<T>/
      dist.png
      gantt_sol<i>.png   (only for solutions that hit the deadline)
      summary.json
"""

import os
import sys
import json
import time
from decimal import Decimal
from collections import Counter

import matplotlib
matplotlib.use("Agg")           # non-interactive backend — safe for scripts
import matplotlib.pyplot as plt

# ── path setup ────────────────────────────────────────────────────────────────
base_dir = os.path.dirname(os.path.abspath(__file__))
workspace_dir = os.path.abspath(os.path.join(base_dir, "../.."))
if workspace_dir not in sys.path:
    sys.path.insert(0, workspace_dir)

from solver_base import (
    read_json,
    build_predecessors,
    infer_activity_states_without_state_file,
    build_reference_no_crash_schedule,
    SolveConfig,
    build_model_and_solve,
    generate_gantt_comparison_plot,
)

# ── helpers ───────────────────────────────────────────────────────────────────

def find_all_diophantine_solutions(activity_data, target_cost, scale=10):
    """Exact backtracking: find all (c_i) s.t. sum(cost_i * c_i) == target_cost."""
    activities = list(activity_data.keys())
    active_tasks = []
    for a in activities:
        nt = int(activity_data[a]["activity_normal_time"])
        mt = int(activity_data[a]["activity_min_time"])
        max_c = nt - mt
        cost = int(Decimal(str(activity_data[a].get("crash_cost", 0.0))) * scale)
        if max_c > 0 and cost > 0:
            active_tasks.append((a, cost, max_c))

    active_tasks.sort(key=lambda x: x[1], reverse=True)
    target_scaled = int(Decimal(str(target_cost)) * scale)
    solutions = []

    suffix_max = [0] * (len(active_tasks) + 1)
    for i in range(len(active_tasks) - 1, -1, -1):
        suffix_max[i] = suffix_max[i + 1] + active_tasks[i][1] * active_tasks[i][2]

    def backtrack(idx, cur_cost, cur_sol):
        if cur_cost == target_scaled:
            full = {a: cur_sol.get(a, 0) for a in activities}
            solutions.append(full)
            return
        if idx >= len(active_tasks):
            return
        if cur_cost + suffix_max[idx] < target_scaled:
            return
        name, cost, max_c = active_tasks[idx]
        for c_val in range(max_c + 1):
            nc = cur_cost + c_val * cost
            if nc <= target_scaled:
                cur_sol[name] = c_val
                backtrack(idx + 1, nc, cur_sol)
                del cur_sol[name]
            else:
                break

    backtrack(0, 0, {})
    return solutions


def save_distribution_plot(T, cost, counts, out_path):
    """Save the makespan-frequency bar chart for a given deadline T and cost."""
    plt.figure(figsize=(10, 6), dpi=150)

    try:
        style = "seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default"
        plt.style.use(style)
    except Exception:
        pass

    x_vals = sorted(counts.keys())
    y_vals = [counts[x] for x in x_vals]

    base_color  = "#1e82a8"
    match_color = "#e67e22"   # highlight bars where makespan == T
    colors = [match_color if x == T else base_color for x in x_vals]

    bars = plt.bar(x_vals, y_vals, color=colors, width=0.6, edgecolor="none", zorder=3)

    for bar in bars:
        h = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + max(1, max(y_vals) * 0.02),
            f"{int(h)}",
            ha="center", va="bottom",
            fontsize=9, fontweight="bold", color="#2c3e50",
        )

    plt.title(
        f"Makespan Distribution — Deadline T = {T} days  |  Min Cost = {cost}",
        fontsize=13, fontweight="bold", pad=14, color="#2c3e50",
    )
    plt.xlabel("Project Makespan (Days)", fontsize=11, fontweight="bold", labelpad=8, color="#2c3e50")
    plt.ylabel("Number of Solutions", fontsize=11, fontweight="bold", labelpad=8, color="#2c3e50")
    plt.xticks(x_vals, [f"{x}d" for x in x_vals], fontsize=8, rotation=45 if len(x_vals) > 10 else 0)
    plt.ylim(0, max(y_vals) + max(2, max(y_vals) * 0.15))
    plt.grid(axis="y", linestyle="--", alpha=0.7, zorder=0)
    plt.grid(axis="x", linestyle="")

    # Annotate deadline bar if it exists
    if T in counts:
        plt.annotate(
            f"Deadline met\n(T = {T} days)\n{counts[T]} solution(s)",
            xy=(T, counts[T]),
            xytext=(T + (max(x_vals) - min(x_vals)) * 0.05 + 0.5, counts[T] + max(y_vals) * 0.12),
            arrowprops=dict(facecolor=match_color, shrink=0.08, width=1.5, headwidth=6),
            fontsize=9, fontweight="bold", color=match_color, ha="left",
        )

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  [dist] Saved → {out_path}")


# ── main pipeline ─────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print("PROJECT CRASHING SWEEP — ALL DEADLINES")
    print("=" * 80)

    # ── load data ────────────────────────────────────────────────────────────
    activity_data_path      = os.path.join(base_dir, "../data/activity_data_v3.json")
    resource_capacity_path  = os.path.join(base_dir, "../data/resource_capacity_v3.json")
    resource_req_path       = os.path.join(base_dir, "../data/resource_requirements_v3.json")

    activity_data       = read_json(activity_data_path)
    resource_capacity   = read_json(resource_capacity_path)
    resource_requirements = read_json(resource_req_path)

    activities = list(activity_data.keys())

    predecessors, _ = build_predecessors(activity_data, [], True)

    current_day = 0
    states, _ = infer_activity_states_without_state_file(
        activity_data, resource_requirements, resource_capacity,
        predecessors, current_day, 60.0, 1,
    )

    # Build baseline schedule once (used for Gantt comparison plots)
    baseline_schedule = build_reference_no_crash_schedule(
        activity_data, resource_requirements, resource_capacity,
        predecessors, current_day, 60.0, 1,
    )

    # ── Step 1: normal makespan (no crashing) ────────────────────────────────
    print("\n[Step 1] Solving min-makespan with NO crashing …")
    cfg_normal = SolveConfig(
        target_end_date=9999,
        current_day=current_day,
        time_limit=60.0,
        num_workers=1,
        auto_fix_paint_trim_cycle=True,
        remove_edges=[],
    )
    # Build normal-time only activity_data (min == normal so no crashing possible)
    activity_data_no_crash = {}
    for a in activities:
        nt = int(activity_data[a]["activity_normal_time"])
        activity_data_no_crash[a] = {
            "activity_normal_time": nt,
            "activity_min_time": nt,      # pin to normal → no crash
            "crash_cost": activity_data[a]["crash_cost"],
        }

    res_normal = build_model_and_solve(
        activity_data_no_crash, resource_requirements, resource_capacity,
        predecessors, states, cfg_normal, "min_makespan",
    )
    assert res_normal["status"] in ("OPTIMAL", "FEASIBLE"), "Normal makespan solve failed!"
    T_normal = res_normal["makespan"]
    print(f"  Normal makespan (no crashing) = {T_normal} days")

    # ── Step 2: minimum possible makespan (max crashing) ─────────────────────
    print("\n[Step 2] Solving min-makespan with FULL crashing …")
    cfg_min = SolveConfig(
        target_end_date=9999,
        current_day=current_day,
        time_limit=60.0,
        num_workers=1,
        auto_fix_paint_trim_cycle=True,
        remove_edges=[],
    )
    res_min = build_model_and_solve(
        activity_data, resource_requirements, resource_capacity,
        predecessors, states, cfg_min, "min_makespan",
    )
    assert res_min["status"] in ("OPTIMAL", "FEASIBLE"), "Min makespan solve failed!"
    T_min = res_min["makespan"]
    print(f"  Minimum possible makespan     = {T_min} days")

    print(f"\n→ Will sweep T from {T_normal} down to {T_min}  ({T_normal - T_min + 1} deadline values)\n")

    # ── output directory ─────────────────────────────────────────────────────
    out_root = os.path.join(base_dir, "../outputs/sweep")
    os.makedirs(out_root, exist_ok=True)

    sweep_summary = []

    # ── Step 3: sweep each deadline T ────────────────────────────────────────
    for T in range(T_normal, T_min - 1, -1):
        print("─" * 70)
        print(f"[T = {T}]  Finding minimum crash cost …")

        t_iter_start = time.time()

        # 3a. Find minimum cost for deadline T
        cfg_cost = SolveConfig(
            target_end_date=T,
            current_day=current_day,
            time_limit=60.0,
            num_workers=1,
            auto_fix_paint_trim_cycle=True,
            remove_edges=[],
        )
        res_cost = build_model_and_solve(
            activity_data, resource_requirements, resource_capacity,
            predecessors, states, cfg_cost, "cost_with_deadline",
        )

        if res_cost["status"] not in ("OPTIMAL", "FEASIBLE"):
            print(f"  ✗ Infeasible for T={T}, skipping.")
            sweep_summary.append({"T": T, "status": "INFEASIBLE"})
            continue

        min_cost = round(float(res_cost["total_crash_cost"]), 4)
        print(f"  Minimum crash cost for T={T} → {min_cost}")

        if min_cost == 0.0:
            print(f"  Cost = 0: no crashing needed at T={T}. Only one trivial solution exists.")
            # Distribution plot: just one solution, makespan = T_normal (or whatever it lands)
            out_dir_T = os.path.join(out_root, f"T_{T}")
            os.makedirs(out_dir_T, exist_ok=True)
            dist_path = os.path.join(out_dir_T, "dist.png")
            save_distribution_plot(T, min_cost, Counter({T_normal: 1}), dist_path)
            sweep_summary.append({
                "T": T, "status": "OK", "min_cost": min_cost,
                "num_diophantine": 1, "num_meeting_deadline": 1,
                "gantt_files": [],
            })
            continue

        # 3b. Find all Diophantine solutions for this cost
        print(f"  Enumerating Diophantine solutions for cost={min_cost} …")
        t0 = time.time()
        solutions = find_all_diophantine_solutions(activity_data, min_cost, scale=10)
        print(f"  Found {len(solutions)} mathematical solutions in {time.time()-t0:.2f}s")

        # 3c. Evaluate makespan for each solution
        print(f"  Evaluating makespan for each solution (CP-SAT) …")
        makespans = []
        schedule_results = []   # (idx, makespan, schedule_rows, sol)

        t0 = time.time()
        for sol_idx, sol in enumerate(solutions):
            act_data_mod = {}
            for a in activities:
                nt = int(activity_data[a]["activity_normal_time"])
                crash_days = sol[a]
                act_data_mod[a] = {
                    "activity_normal_time": nt - crash_days,
                    "activity_min_time": nt - crash_days,
                    "crash_cost": activity_data[a]["crash_cost"],
                }

            cfg_eval = SolveConfig(
                target_end_date=9999,
                current_day=current_day,
                time_limit=5.0,
                num_workers=1,
                auto_fix_paint_trim_cycle=True,
                remove_edges=[],
            )
            res_eval = build_model_and_solve(
                act_data_mod, resource_requirements, resource_capacity,
                predecessors, states, cfg_eval, "min_makespan",
            )

            if res_eval["status"] in ("OPTIMAL", "FEASIBLE"):
                ms = res_eval["makespan"]
                makespans.append(ms)
                schedule_results.append((sol_idx, ms, res_eval["schedule"], sol))
            else:
                makespans.append(-1)

        print(f"  Evaluated in {time.time()-t0:.2f}s")

        # 3d. Build frequency counter
        valid_makespans = [m for m in makespans if m > 0]
        counts = Counter(valid_makespans)

        print("  Makespan frequencies:")
        for k in sorted(counts.keys()):
            marker = " ← DEADLINE MET" if k == T else ""
            print(f"    Makespan {k}: {counts[k]} solution(s){marker}")

        # 3e. Save distribution plot immediately
        out_dir_T = os.path.join(out_root, f"T_{T}")
        os.makedirs(out_dir_T, exist_ok=True)
        dist_path = os.path.join(out_dir_T, "dist.png")
        save_distribution_plot(T, min_cost, counts, dist_path)

        # 3f. For each solution that meets the deadline, save a Gantt chart
        matching = [(i, ms, sched, sol) for (i, ms, sched, sol) in schedule_results if ms <= T]
        gantt_files = []

        print(f"  Solutions meeting deadline (makespan ≤ {T}): {len(matching)}")

        for rank, (sol_idx, ms, sched, sol) in enumerate(matching, start=1):
            gantt_name = f"gantt_sol{rank:03d}_ms{ms}.png"
            gantt_path = os.path.join(out_dir_T, gantt_name)

            # Build a modified activity_data to pass to the Gantt generator
            # (so crash_days are shown correctly based on the original normal times)
            # We need to inject crash_days back into the schedule rows.
            # The schedule rows from min_makespan already have crash_days = solver.Value(c[a]),
            # but in our case NT==MT==crashed value, so crash_days will always be 0 there.
            # Instead, we recalculate crash_days from the solution dict:
            for row in sched:
                a = row["activity"]
                row["crash_days"] = sol.get(a, 0)

            try:
                generate_gantt_comparison_plot(
                    baseline_schedule=baseline_schedule,
                    optimized_schedule=sched,
                    current_day=current_day,
                    output_path=gantt_path,
                )
                gantt_files.append(gantt_name)
                print(f"    [gantt] Saved sol {rank} (makespan={ms}) → {gantt_path}")
            except Exception as exc:
                print(f"    [gantt] WARNING: Failed for sol {rank}: {exc}")

        # 3g. Save per-T summary JSON
        summary_T = {
            "T": T,
            "status": "OK",
            "min_cost": min_cost,
            "num_diophantine": len(solutions),
            "makespan_frequencies": {str(k): v for k, v in sorted(counts.items())},
            "num_meeting_deadline": len(matching),
            "gantt_files": gantt_files,
            "elapsed_seconds": round(time.time() - t_iter_start, 2),
        }
        with open(os.path.join(out_dir_T, "summary.json"), "w") as f:
            json.dump(summary_T, f, indent=2)

        sweep_summary.append(summary_T)
        print(f"  ✓ Done T={T} in {summary_T['elapsed_seconds']}s")

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("SWEEP COMPLETE")
    print("=" * 80)
    print(f"{'T':>6}  {'Min Cost':>10}  {'Solutions':>10}  {'Deadline Met':>13}")
    print("-" * 50)
    for entry in sweep_summary:
        if entry.get("status") == "INFEASIBLE":
            print(f"{entry['T']:>6}  {'—':>10}  {'INFEASIBLE':>10}  {'—':>13}")
        else:
            print(
                f"{entry['T']:>6}  {entry.get('min_cost', 0):>10}  "
                f"{entry.get('num_diophantine', 0):>10}  "
                f"{entry.get('num_meeting_deadline', 0):>13}"
            )

    # Save global summary
    global_summary_path = os.path.join(out_root, "sweep_summary.json")
    with open(global_summary_path, "w") as f:
        json.dump(
            {"T_normal": T_normal, "T_min": T_min, "results": sweep_summary},
            f, indent=2,
        )
    print(f"\nGlobal summary saved → {global_summary_path}")
    print(f"All outputs in       → {out_root}")


if __name__ == "__main__":
    main()
