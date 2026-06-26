#!/usr/bin/env python3
import os
import sys
import time
from collections import Counter
import matplotlib.pyplot as plt
from decimal import Decimal

# Add workspace directory to path
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
    build_model_and_solve
)

def find_all_diophantine_solutions(activity_data, target_cost, scale=10):
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
    target_cost_scaled = int(Decimal(str(target_cost)) * scale)
    solutions = []
    
    suffix_max_cost = [0] * (len(active_tasks) + 1)
    for i in range(len(active_tasks) - 1, -1, -1):
        suffix_max_cost[i] = suffix_max_cost[i+1] + active_tasks[i][1] * active_tasks[i][2]
        
    def backtrack(idx, current_cost, current_solution):
        if current_cost == target_cost_scaled:
            full_sol = {a: current_solution.get(a, 0) for a in activities}
            solutions.append(full_sol)
            return
        if idx >= len(active_tasks):
            return
        if current_cost + suffix_max_cost[idx] < target_cost_scaled:
            return
        name, cost, max_c = active_tasks[idx]
        for c_val in range(max_c + 1):
            next_cost = current_cost + c_val * cost
            if next_cost <= target_cost_scaled:
                current_solution[name] = c_val
                backtrack(idx + 1, next_cost, current_solution)
                del current_solution[name]
            else:
                break
                
    backtrack(0, 0, {})
    return solutions

def main():
    print("=" * 80)
    print("GENERATING MAKESPAN DISTRIBUTIONS FOR ALL DEADLINES")
    print("=" * 80)
    
    activity_data_path = os.path.join(base_dir, "../data/activity_data_v3.json")
    resource_capacity_path = os.path.join(base_dir, "../data/resource_capacity_v3.json")
    resource_req_path = os.path.join(base_dir, "../data/resource_requirements_v3.json")
    
    activity_data = read_json(activity_data_path)
    resource_capacity = read_json(resource_capacity_path)
    resource_requirements = read_json(resource_req_path)
    
    activities = list(activity_data.keys())
    predecessors, _ = build_predecessors(activity_data, [], True)
    current_day = 0
    states, _ = infer_activity_states_without_state_file(
        activity_data, resource_requirements, resource_capacity, predecessors, current_day, 60.0, 1
    )
    
    # 1. Find normal makespan (0 crashing)
    baseline_schedule = build_reference_no_crash_schedule(
        activity_data, resource_requirements, resource_capacity, predecessors, current_day, 60.0, 1
    )
    d_norm = max(b["end"] for b in baseline_schedule.values())
    print(f"Normal Makespan (0 crashing): {d_norm} days")
    
    # 2. Find minimum possible makespan
    cfg_min = SolveConfig(
        target_end_date=None,
        current_day=current_day,
        time_limit=60.0,
        num_workers=8,
        auto_fix_paint_trim_cycle=True,
        remove_edges=[]
    )
    res_min = build_model_and_solve(
        activity_data, resource_requirements, resource_capacity, predecessors, states, cfg_min, "min_makespan"
    )
    if res_min.get("status") not in ("OPTIMAL", "FEASIBLE"):
        print("Could not find minimum makespan!")
        return
    d_min = res_min["makespan"]
    print(f"Minimum Makespan: {d_min} days")
    
    out_dir = os.path.join(base_dir, "../outputs/makespan_distributions")
    os.makedirs(out_dir, exist_ok=True)
    
    # Track costs we've already processed to avoid duplicating work if multiple deadlines give the same minimum cost
    processed_costs = set()
    
    # 3. Iterate deadlines
    for d in range(d_norm, d_min - 1, -1):
        print("-" * 40)
        print(f"Processing Deadline: {d}")
        
        cfg_cost = SolveConfig(
            target_end_date=d,
            current_day=current_day,
            time_limit=60.0,
            num_workers=8,
            auto_fix_paint_trim_cycle=True,
            remove_edges=[]
        )
        res_cost = build_model_and_solve(
            activity_data, resource_requirements, resource_capacity, predecessors, states, cfg_cost, "cost_with_deadline"
        )
        
        if res_cost.get("status") not in ("OPTIMAL", "FEASIBLE"):
            print(f"Deadline {d} is infeasible!")
            continue
            
        min_cost = res_cost["objective_value"]
        print(f"  Minimum Cost for Deadline {d}: {min_cost}")
        
        if min_cost in processed_costs:
            print(f"  Cost {min_cost} already processed for a previous deadline. Skipping plot generation.")
            continue
            
        processed_costs.add(min_cost)
        
        diophantine_solutions = find_all_diophantine_solutions(activity_data, min_cost)
        print(f"  Found {len(diophantine_solutions)} Diophantine solutions for cost {min_cost}.")
        
        if len(diophantine_solutions) == 0:
            continue
            
        makespans = []
        for sol in diophantine_solutions:
            act_data_mod = {}
            for a in activities:
                nt = int(activity_data[a]["activity_normal_time"])
                crash_days = sol[a]
                act_data_mod[a] = {
                    "activity_normal_time": nt - crash_days,
                    "activity_min_time": nt - crash_days,
                    "crash_cost": activity_data[a]["crash_cost"]
                }
                
            cfg = SolveConfig(
                target_end_date=max(d_norm, 300),  # High enough
                current_day=current_day,
                time_limit=5.0,
                num_workers=1,
                auto_fix_paint_trim_cycle=True,
                remove_edges=[]
            )
            
            res = build_model_and_solve(
                act_data_mod, resource_requirements, resource_capacity, predecessors, states, cfg, "min_makespan"
            )
            
            if res.get("status") in ("OPTIMAL", "FEASIBLE"):
                makespans.append(res.get("makespan"))
            else:
                makespans.append(-1)
                
        valid_makespans = [m for m in makespans if m > 0]
        counts = Counter(valid_makespans)
        
        # Plot
        plt.figure(figsize=(10, 6), dpi=150)
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        
        x_vals = sorted(counts.keys())
        y_vals = [counts[x] for x in x_vals]
        
        color = "#1e82a8"
        hover_color = "#e67e22"
        # highlight the specific deadline `d` if it's in the solutions
        colors = [hover_color if x == d else color for x in x_vals]
        
        bars = plt.bar(x_vals, y_vals, color=colors, width=0.6, edgecolor="none", zorder=3)
        
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.1 * max(y_vals, default=0),
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                color="#2c3e50"
            )
            
        plt.title(f"Distribution of Makespans for Target Cost = {min_cost}\n(Minimum Cost for Deadline = {d} Days)", fontsize=14, fontweight="bold", pad=15, color="#2c3e50")
        plt.xlabel("Project Makespan (Days)", fontsize=11, fontweight="bold", labelpad=10, color="#2c3e50")
        plt.ylabel("Number of Solutions", fontsize=11, fontweight="bold", labelpad=10, color="#2c3e50")
        
        if len(x_vals) <= 20:
            plt.xticks(x_vals, [f"{x} days" for x in x_vals], fontsize=9, rotation=45 if len(x_vals) > 10 else 0)
        
        plt.ylim(0, max(y_vals) * 1.2 if y_vals else 10)
        plt.grid(axis="y", linestyle="--", alpha=0.7, zorder=0)
        plt.grid(axis="x", linestyle="")
        
        if d in counts:
            plt.annotate(
                f"Deadline {d}",
                xy=(d, counts[d]),
                xytext=(d, counts[d] + 0.15 * max(y_vals)),
                arrowprops=dict(facecolor="#e67e22", shrink=0.08, width=1.5, headwidth=6),
                fontsize=10,
                fontweight="bold",
                color="#e67e22",
                ha="center"
            )
            
        plt.tight_layout()
        
        out_path = os.path.join(out_dir, f"makespan_distribution_deadline_{d}.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"  Saved plot to {out_path}")

if __name__ == "__main__":
    main()
