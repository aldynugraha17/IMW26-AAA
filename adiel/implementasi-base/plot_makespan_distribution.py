#!/usr/bin/env python3
import os
import sys
import json
import time
from decimal import Decimal
from collections import Counter
import matplotlib.pyplot as plt

# Add workspace directory to path
base_dir = os.path.dirname(os.path.abspath(__file__))
workspace_dir = os.path.abspath(os.path.join(base_dir, "../.."))
if workspace_dir not in sys.path:
    sys.path.insert(0, workspace_dir)

from solver_base import (
    read_json,
    build_predecessors,
    infer_activity_states_without_state_file,
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
    print("GENERATING MAKESPAN DISTRIBUTION PLOT")
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
    
    # 1. Find all 346 diophantine solutions for cost = 180
    diophantine_solutions = find_all_diophantine_solutions(activity_data, 180.0)
    print(f"Found {len(diophantine_solutions)} Diophantine solutions.")
    
    # 2. Evaluate makespan for each solution using CP-SAT
    print("Evaluating makespan for each solution...")
    makespans = []
    t0 = time.time()
    
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
            target_end_date=255,  # High enough to always be feasible
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
            makespans.append(-1)  # error/infeasible marker
            
    t1 = time.time()
    print(f"Evaluated all solutions in {t1 - t0:.2f} seconds.")
    
    # Filter valid makespans
    valid_makespans = [m for m in makespans if m > 0]
    counts = Counter(valid_makespans)
    
    print("\nMakespan frequencies:")
    for k in sorted(counts.keys()):
        print(f"  Makespan {k}: {counts[k]} solutions")
        
    # 3. Create the distribution plot
    plt.figure(figsize=(10, 6), dpi=150)
    
    # Minimalist styling
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    
    x_vals = sorted(counts.keys())
    y_vals = [counts[x] for x in x_vals]
    
    # Vibrant design colors (Teal-blue)
    color = "#1e82a8"
    hover_color = "#e67e22" # Highlight the unique 243 optimal solution
    colors = [hover_color if x == 243 else color for x in x_vals]
    
    bars = plt.bar(x_vals, y_vals, color=colors, width=0.6, edgecolor="none", zorder=3)
    
    # Add values on top of each bar
    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 2,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
            color="#2c3e50"
        )
        
    plt.title("Distribution of Makespans for Diophantine Solutions\n(Target Cost = 180.0)", fontsize=14, fontweight="bold", pad=15, color="#2c3e50")
    plt.xlabel("Project Makespan (Days)", fontsize=11, fontweight="bold", labelpad=10, color="#2c3e50")
    plt.ylabel("Number of Solutions", fontsize=11, fontweight="bold", labelpad=10, color="#2c3e50")
    plt.xticks(x_vals, [f"{x} days" for x in x_vals], fontsize=9)
    plt.ylim(0, max(y_vals) + 12)
    plt.grid(axis="y", linestyle="--", alpha=0.7, zorder=0)
    plt.grid(axis="x", linestyle="")
    
    # Highlight the unique 243 solution
    plt.annotate(
        "Unique Feasible Schedule\n(Makespan = 243 days)",
        xy=(243, 1),
        xytext=(242.5, 30),
        arrowprops=dict(facecolor="#e67e22", shrink=0.08, width=1.5, headwidth=6),
        fontsize=10,
        fontweight="bold",
        color="#e67e22",
        ha="center"
    )
    
    plt.tight_layout()
    
    # Save the plot
    out_dir = os.path.join(base_dir, "../outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "makespan_distribution_180.png")
    plt.savefig(out_path, dpi=300)
    print(f"\nPlot saved to {out_path}")
    
    # Also save to conversation artifacts directory if it exists
    artifact_dir = "/Users/macintoshhd/.gemini/antigravity-cli/brain/b9bf883f-976b-4066-b4be-1515c9c978ef"
    if os.path.exists(artifact_dir):
        artifact_path = os.path.join(artifact_dir, "makespan_distribution_180.png")
        plt.savefig(artifact_path, dpi=300)
        print(f"Plot copied to artifact dir: {artifact_path}")

if __name__ == "__main__":
    main()
