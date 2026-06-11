import os
import json
from solver_base import (
    read_json,
    build_predecessors,
    infer_activity_states_without_state_file,
    SolveConfig,
    build_model_and_solve,
    write_json,
    write_schedule_csv
)

def run_bonus_penalty():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    activity_data_path = os.path.join(base_dir, "../data/activity_data_v3.json")
    resource_capacity_path = os.path.join(base_dir, "../data/resource_capacity_v3.json")
    resource_req_path = os.path.join(base_dir, "../data/resource_requirements_v3.json")
    
    activity_data = read_json(activity_data_path)
    resource_capacity = read_json(resource_capacity_path)
    resource_requirements = read_json(resource_req_path)
    
    predecessors, _ = build_predecessors(activity_data, [], True)
    
    current_day = 0
    target_end_date = 243
    c_late = 1000.0
    c_early = 500.0
    
    states, _ = infer_activity_states_without_state_file(
        activity_data, resource_requirements, resource_capacity,
        predecessors, current_day, 60.0, 1
    )
    
    cfg = SolveConfig(
        target_end_date=target_end_date,
        current_day=current_day,
        time_limit=60.0,
        num_workers=1,
        auto_fix_paint_trim_cycle=True,
        remove_edges=[],
        c_late=c_late,
        c_early=c_early
    )
    
    result = build_model_and_solve(
        activity_data,
        resource_requirements,
        resource_capacity,
        predecessors,
        states,
        cfg,
        mode="bonus_penalty",
    )
    
    print("Bonus-Penalty Result:")
    print("Status:", result["status"])
    if "makespan" in result:
        print("Makespan:", result["makespan"])
        if result["makespan"] > target_end_date:
            print("Penalty applied:", (result["makespan"] - target_end_date) * c_late)
        elif result["makespan"] < target_end_date:
            print("Bonus applied:", (target_end_date - result["makespan"]) * c_early)
            
    if "total_crash_cost" in result:
        print("Total crash cost:", result["total_crash_cost"])
        
    out_dir = os.path.join(base_dir, "../outputs")
    os.makedirs(out_dir, exist_ok=True)
    write_json(os.path.join(out_dir, "base_bonus_penalty.json"), result)
    if "schedule" in result:
        write_schedule_csv(os.path.join(out_dir, "base_bonus_penalty_schedule.csv"), result["schedule"])

if __name__ == "__main__":
    run_bonus_penalty()
