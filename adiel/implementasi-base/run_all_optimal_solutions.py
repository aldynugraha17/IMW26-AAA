#!/usr/bin/env python3
import os
import sys
import json
import time
from decimal import Decimal

# Add the workspace directory to the path so we can import from aldy if needed
base_dir = os.path.dirname(os.path.abspath(__file__))
workspace_dir = os.path.abspath(os.path.join(base_dir, "../.."))
if workspace_dir not in sys.path:
    sys.path.insert(0, workspace_dir)

from solver_base import (
    read_json,
    build_predecessors,
    infer_activity_states_without_state_file,
    SolveConfig,
    build_model_and_solve,
    write_json
)

# Try to import from aldy.SPOC_fullcode_raw
try:
    from aldy.SPOC_fullcode_raw import solve_diophantine_soac, problem_1
    HAS_SOAC = True
except ImportError:
    HAS_SOAC = False

def find_all_diophantine_solutions(activity_data, target_cost, scale):
    """
    Finds all integer solutions to the Diophantine equation:
        sum(crash_cost_i * c_i) == target_cost
    using an exact recursive backtracking search (highly optimized).
    """
    activities = list(activity_data.keys())
    active_tasks = []
    
    for a in activities:
        nt = int(activity_data[a]["activity_normal_time"])
        mt = int(activity_data[a]["activity_min_time"])
        max_c = nt - mt
        cost = int(Decimal(str(activity_data[a].get("crash_cost", 0.0))) * scale)
        if max_c > 0 and cost > 0:
            active_tasks.append((a, cost, max_c))
            
    # Sort by cost descending for efficient branch pruning
    active_tasks.sort(key=lambda x: x[1], reverse=True)
    
    target_cost_scaled = int(Decimal(str(target_cost)) * scale)
    solutions = []
    
    # Precompute suffix maximum possible costs to prune unfeasible branches early
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

def run_pipeline(method="backtrack"):
    print("=" * 80)
    print("PROJECT CRASHING - ALL OPTIMAL SOLUTIONS FINDER")
    print(f"Method for Diophantine Equation: {method.upper()}")
    print("=" * 80)
    
    activity_data_path = os.path.join(base_dir, "../data/activity_data_v3.json")
    resource_capacity_path = os.path.join(base_dir, "../data/resource_capacity_v3.json")
    resource_req_path = os.path.join(base_dir, "../data/resource_requirements_v3.json")
    
    activity_data = read_json(activity_data_path)
    resource_capacity = read_json(resource_capacity_path)
    resource_requirements = read_json(resource_req_path)
    
    activities = list(activity_data.keys())
    
    predecessors, _ = build_predecessors(
        activity_data=activity_data,
        remove_edges=[],
        auto_fix_paint_trim_cycle=True,
    )
    
    current_day = 0
    target_end_date = 243
    target_cost = 180.0
    scale = 10  # cost scaling factor (e.g. 60.0 -> 600)
    
    states, _ = infer_activity_states_without_state_file(
        activity_data=activity_data,
        resource_requirements=resource_requirements,
        resource_capacity=resource_capacity,
        predecessors=predecessors,
        current_day=current_day,
        time_limit=60.0,
        num_workers=1,
    )
    
    # Step 1: Find all integer solutions to the Diophantine Equation
    t0 = time.time()
    diophantine_solutions = []
    
    if method == "soac":
        if not HAS_SOAC:
            print("Error: Could not import SOAC library from aldy.SPOC_fullcode_raw.py")
            sys.exit(1)
        print("Running SOAC Diophantine solver (SDOA + Clustering)... This might take a few minutes...")
        roots = solve_diophantine_soac(problem_1, sort_solutions=False)
        # Convert roots (task durations) back to crash amounts c_i = NT_i - d_i
        for root in roots:
            sol = {}
            for i, a in enumerate(activities):
                nt = int(activity_data[a]["activity_normal_time"])
                sol[a] = nt - root[i]
            diophantine_solutions.append(sol)
    else:
        print("Running Exact Backtracking Diophantine solver...")
        diophantine_solutions = find_all_diophantine_solutions(activity_data, target_cost, scale)
        
    t_diophantine = time.time() - t0
    print(f"Found {len(diophantine_solutions)} Diophantine solutions in {t_diophantine:.3f} seconds.")
    
    # Step 2: Verify each solution against scheduling/resource constraints in CP-SAT
    print("\nVerifying solutions using CP-SAT...")
    valid_schedules = []
    t_verify_start = time.time()
    
    for idx, sol in enumerate(diophantine_solutions):
        # Create a modified copy of activity_data where normal & min duration are fixed to the crashed value
        act_data_mod = {}
        for a in activities:
            nt = int(activity_data[a]["activity_normal_time"])
            crash_days = sol[a]
            act_data_mod[a] = {
                "activity_normal_time": nt - crash_days,
                "activity_min_time": nt - crash_days,
                "crash_cost": activity_data[a]["crash_cost"],
                "required_activities": activity_data[a]["required_activities"]
            }
            
        cfg = SolveConfig(
            target_end_date=target_end_date,
            current_day=current_day,
            time_limit=5.0,
            num_workers=1,
            auto_fix_paint_trim_cycle=True,
            remove_edges=[]
        )
        
        # Run CP-SAT in min_makespan mode
        result = build_model_and_solve(
            act_data_mod,
            resource_requirements,
            resource_capacity,
            predecessors,
            states,
            cfg,
            mode="min_makespan"
        )
        
        if result.get("status") in ("OPTIMAL", "FEASIBLE"):
            makespan = result.get("makespan", 999)
            if makespan <= target_end_date:
                # Format crashed activities list
                crashed_list = []
                for a in activities:
                    if sol[a] > 0:
                        crashed_list.append({
                            "activity": a,
                            "crash_days": sol[a],
                            "crash_cost_per_day": activity_data[a]["crash_cost"],
                            "total_cost": float(Decimal(str(activity_data[a]["crash_cost"])) * Decimal(sol[a]))
                        })
                
                valid_schedules.append({
                    "solution_index": idx + 1,
                    "makespan": makespan,
                    "total_crash_cost": target_cost,
                    "crashed_activities": crashed_list,
                    "schedule": result["schedule"]
                })
                
    t_verify = time.time() - t_verify_start
    print(f"Verification of all solutions completed in {t_verify:.3f} seconds.")
    print(f"Number of feasible schedules under deadline {target_end_date}: {len(valid_schedules)}")
    print("-" * 80)
    
    # Save results to outputs
    out_dir = os.path.join(base_dir, "../outputs")
    os.makedirs(out_dir, exist_ok=True)
    
    output_payload = {
        "status": "SUCCESS",
        "method": method,
        "target_end_date": target_end_date,
        "target_cost": target_cost,
        "diophantine_solutions_count": len(diophantine_solutions),
        "feasible_schedules_count": len(valid_schedules),
        "feasible_schedules": valid_schedules
    }
    
    # Save the feasibility results
    write_json(os.path.join(out_dir, "all_optimal_solutions.json"), output_payload)
    print(f"Results saved to outputs/all_optimal_solutions.json")
    
    # Save all mathematical diophantine solutions (crashes > 0)
    diophantine_payload = [
        {a: sol[a] for a in activities if sol[a] > 0}
        for sol in diophantine_solutions
    ]
    write_json(os.path.join(out_dir, "all_diophantine_solutions.json"), diophantine_payload)
    print(f"All mathematical diophantine solutions saved to outputs/all_diophantine_solutions.json")
    
    # Print summary of feasible schedules
    for i, s in enumerate(valid_schedules):
        crashes_desc = ", ".join([f"'{c['activity']}': {c['crash_days']}" for c in s["crashed_activities"]])
        print(f"Schedule {i+1} (Index {s['solution_index']}):")
        print(f"  • Crashes  : {{{crashes_desc}}}")
        print(f"  • Makespan : {s['makespan']}")
    print("=" * 80)

if __name__ == "__main__":
    method = "backtrack"
    if len(sys.argv) > 1 and sys.argv[1] in ("soac", "backtrack"):
        method = sys.argv[1]
    run_pipeline(method)
