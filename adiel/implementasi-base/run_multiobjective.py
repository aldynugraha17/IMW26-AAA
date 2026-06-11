import os
import json
import matplotlib.pyplot as plt
from solver_base import (
    read_json,
    build_predecessors,
    infer_activity_states_without_state_file,
    SolveConfig,
    build_model_and_solve
)

def run_multiobjective():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    activity_data_path = os.path.join(base_dir, "../data/activity_data_v3.json")
    resource_capacity_path = os.path.join(base_dir, "../data/resource_capacity_v3.json")
    resource_req_path = os.path.join(base_dir, "../data/resource_requirements_v3.json")
    
    activity_data = read_json(activity_data_path)
    resource_capacity = read_json(resource_capacity_path)
    resource_requirements = read_json(resource_req_path)
    
    predecessors, _ = build_predecessors(activity_data, [], True)
    
    current_day = 0
    states, _ = infer_activity_states_without_state_file(
        activity_data, resource_requirements, resource_capacity,
        predecessors, current_day, 60.0, 1
    )
    
    # normal duration (min makespan without crashing)
    cfg_normal = SolveConfig(
        target_end_date=None, current_day=current_day, time_limit=60.0,
        num_workers=1, auto_fix_paint_trim_cycle=True, remove_edges=[]
    )
    res_normal = build_model_and_solve(
        activity_data, resource_requirements, resource_capacity,
        predecessors, states, cfg_normal, mode="min_makespan"
    )
    normal_duration = res_normal.get("makespan", 250)
    
    # max crashable duration (min makespan with max budget/all crashed)
    # we can just run min_makespan and allow any cost, wait min_makespan does not crash because it minimizes Cmax only? 
    # Actually, min_makespan only minimizes Cmax, so it WILL crash if it reduces Cmax. 
    # Ah, in solver_base.py:
    # else:  # min_makespan
    #    model.Minimize(Cmax)
    # This will crash tasks as much as possible if it reduces Cmax. Because crash cost is NOT in objective.
    max_crashed_duration = res_normal.get("makespan", 200)
    
    # Wait, the reference no crash schedule has normal duration.
    from solver_base import build_reference_no_crash_schedule
    baseline = build_reference_no_crash_schedule(
        activity_data, resource_requirements, resource_capacity,
        predecessors, current_day, 60.0, 1
    )
    normal_duration = max(row["end"] for row in baseline.values())
    
    print(f"Normal Duration: {normal_duration}")
    print(f"Max Crashed Duration: {max_crashed_duration}")
    
    points = []
    
    for t in range(int(max_crashed_duration), int(normal_duration) + 1, max(1, (int(normal_duration) - int(max_crashed_duration)) // 10)):
        print(f"Solving for Target Deadline = {t}")
        cfg = SolveConfig(
            target_end_date=t, current_day=current_day, time_limit=30.0,
            num_workers=1, auto_fix_paint_trim_cycle=True, remove_edges=[]
        )
        res = build_model_and_solve(
            activity_data, resource_requirements, resource_capacity,
            predecessors, states, cfg, mode="cost_with_deadline"
        )
        if res["status"] in ["OPTIMAL", "FEASIBLE"]:
            cost = res.get("total_crash_cost", 0)
            makespan = res.get("makespan", t)
            points.append((makespan, cost))
            print(f"  -> Makespan: {makespan}, Cost: {cost}")
        else:
            print(f"  -> {res['status']}")
            
    # plot Pareto front
    if points:
        points.sort()
        ms = [p[0] for p in points]
        c = [p[1] for p in points]
        
        plt.figure(figsize=(8, 6))
        plt.plot(ms, c, marker='o', linestyle='-', color='b')
        plt.xlabel('Makespan (days)')
        plt.ylabel('Total Crash Cost ($)')
        plt.title('Time-Cost Pareto Front (Base Model)')
        plt.grid(True)
        
        out_dir = os.path.join(base_dir, "../outputs")
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(os.path.join(out_dir, "base_pareto_front.png"))
        print(f"Saved Pareto front plot to {os.path.join(out_dir, 'base_pareto_front.png')}")

if __name__ == "__main__":
    run_multiobjective()
