#!/usr/bin/env python3
"""Rebuild <out_dir>/_paths/world_XX.json from an existing run's episodes.json, so a
rerun of run_global_inference.py reuses those frozen RRT* paths instead of
replanning.

The plan stage in run_global_inference.py (see paths_file() / main()) skips a world
whenever <out_dir>/_paths/world_{widx:02d}.json already exists, unless --fresh is
passed. This script writes exactly those files, in the exact schema stage_plan()
writes them in, sourced from a previous run's committed episodes.json instead of a
fresh OMPL solve.

Usage:
    python inference/freeze_paths.py \
        --from inference/results/global_paired/20260616_002825/episodes.json \
        --to   inference/results/global_paired/<new_run_dir>

Then run run_global_inference.py with --output_dir pointing at <new_run_dir> and
WITHOUT --fresh: stage 1 (planning) will report 0 tasks and skip straight to the
rollouts, using these frozen paths.
"""
import argparse
import json
import os

# Same fields stage_plan() writes per episode into _paths/world_XX.json, minus
# controller_results (that belongs to episodes.json / _ctrl, not to the plan cache).
PATH_FIELDS = [
    "episode_id", "world_id", "difficulty", "start_pose", "goal_pose",
    "straight_line_dist", "planning_success", "planner_path",
    "planner_path_length", "planner_time_ms", "planner_path_geometry",
    "planner_metadata",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="src", required=True,
                     help="episodes.json of the run whose paths should be reused")
    ap.add_argument("--to", dest="out_dir", required=True,
                     help="output_dir the new run_global_inference.py call will use")
    ap.add_argument("--eps_per_world", type=int, default=None,
                     help="defaults to config['n_episodes_per_world'] in --from")
    args = ap.parse_args()

    with open(args.src) as f:
        data = json.load(f)
    episodes = data["episodes"]
    eps_per_world = args.eps_per_world or data["config"]["n_episodes_per_world"]

    by_world = {}
    for ep in episodes:
        widx = ep["episode_id"] // eps_per_world
        rec = {k: ep[k] for k in PATH_FIELDS if k in ep}
        by_world.setdefault(widx, []).append(rec)

    paths_dir = os.path.join(args.out_dir, "_paths")
    os.makedirs(paths_dir, exist_ok=True)

    for widx, recs in sorted(by_world.items()):
        recs.sort(key=lambda r: r["episode_id"])
        out_path = os.path.join(paths_dir, f"world_{widx:02d}.json")
        tmp = out_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"world_idx": widx, "episodes": recs}, f)
        os.replace(tmp, out_path)
        n_ok = sum(1 for r in recs if r.get("planning_success"))
        print(f"world_{widx:02d}: {len(recs)} episodes ({n_ok} planned) -> {out_path}")

    print(f"\n{len(by_world)} worlds frozen into {paths_dir}")
    print("Rerun run_global_inference.py with --output_dir here and NO --fresh: "
          "stage 1 will report 0 plan tasks.")


if __name__ == "__main__":
    main()
