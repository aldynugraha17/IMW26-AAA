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

def run_time_driven():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    activity_data_path = os.path.join(base_dir, "../data/activity_data_v3.json")
    resource_capacity_path = os.path.join(base_dir, "../data/resource_capacity_v3.json")
    resource_req_path = os.path.join(base_dir, "../data/resource_requirements_v3.json")
    
    activity_data = read_json(activity_data_path)
    resource_capacity = read_json(resource_capacity_path)
    resource_requirements = read_json(resource_req_path)
    
    predecessors, _ = build_predecessors(activity_data, [], True)
    
    current_day = 0
    budget_limit = 5000.0
    
    states, _ = infer_activity_states_without_state_file(
        activity_data, resource_requirements, resource_capacity,
        predecessors, current_day, 60.0, 1
    )
    
    cfg = SolveConfig(
        target_end_date=None,
        current_day=current_day,
        time_limit=60.0,
        num_workers=1,
        auto_fix_paint_trim_cycle=True,
        remove_edges=[],
        budget_limit=budget_limit
    )
    
    result = build_model_and_solve(
        activity_data,
        resource_requirements,
        resource_capacity,
        predecessors,
        states,
        cfg,
        mode="time_with_budget",
    )
    
    print("Time Driven Result (Budget Limit = {}):".format(budget_limit))
    print("Status:", result["status"])
    if "makespan" in result:
        print("Makespan:", result["makespan"])
    if "total_crash_cost" in result:
        print("Total crash cost:", result["total_crash_cost"])
        
    out_dir = os.path.join(base_dir, "../outputs")
    os.makedirs(out_dir, exist_ok=True)
    write_json(os.path.join(out_dir, "base_time_driven.json"), result)
    if "schedule" in result:
        write_schedule_csv(os.path.join(out_dir, "base_time_driven_schedule.csv"), result["schedule"])

if __name__ == "__main__":
    run_time_driven()
