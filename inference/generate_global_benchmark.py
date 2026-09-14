#!/usr/bin/env python3
"""
One-time script to generate a fixed long-horizon benchmark set for global (end-to-end) evaluation.

For each test world, iterates seeds until n_episodes valid start/goal pairs are found where
the straight-line distance is in [min_goal_distance, max_goal_distance] meters.

All controllers evaluated with run_global_inference.py will share these exact episodes.

Usage:
    python inference/generate_global_benchmark.py
    python inference/generate_global_benchmark.py --n_episodes 20  # fast validation
"""

import os
import sys
import gc
import json
import argparse
import numpy as np

root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)

from envs.gym_bunker_env import BunkerEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from gymnasium.wrappers import TimeLimit

parser = argparse.ArgumentParser(description="Generate fixed long-horizon benchmark episode set")
parser.add_argument("--n_episodes", type=int, default=100,
                    help="Number of episodes per world (default: 100)")
parser.add_argument("--min_goal_distance", type=float, default=8.0,
                    help="Minimum straight-line start-to-goal distance in metres (default: 8.0)")
parser.add_argument("--max_goal_distance", type=float, default=20.0,
                    help="Maximum straight-line start-to-goal distance in metres (default: 20.0)")
parser.add_argument("--out", type=str, default=None,
                    help="Output path (default: inference/global_benchmark.json)")
args = parser.parse_args()

test_worlds_dir = os.path.join(root, "assets", "worlds", "test")
out_path = args.out or os.path.join(root, "inference", "global_benchmark.json")

# Discover all test worlds (XML files with matching metadata)
world_entries = []
for fname in sorted(os.listdir(test_worlds_dir)):
    if not fname.endswith(".xml"):
        continue
    stem = fname[:-4]
    meta_path = os.path.join(test_worlds_dir, f"{stem}_metadata.json")
    if not os.path.exists(meta_path):
        print(f"  [WARN] No metadata for {fname}, skipping.")
        continue
    with open(meta_path) as f:
        meta = json.load(f)
    difficulty = meta.get("difficulty", "unknown")
    world_entries.append({
        "path": os.path.join("test", stem),
        "xml_path": os.path.join(test_worlds_dir, fname),
        "difficulty": difficulty,
    })

print(f"Found {len(world_entries)} test worlds:")
for w in world_entries:
    print(f"  [{w['difficulty']:6s}]  {w['path']}")

print(f"\nGenerating {args.n_episodes} long-horizon episodes per world "
      f"(goal dist in [{args.min_goal_distance}, {args.max_goal_distance}] m)...\n")

benchmark = {
    "n_episodes_per_world": args.n_episodes,
    "min_goal_distance": args.min_goal_distance,
    "max_goal_distance": args.max_goal_distance,
    "worlds": [],
}

for w in world_entries:
    print(f"  World: {w['path']} ({w['difficulty']}) ...", flush=True)

    # max_goal_sampling_distance=20 allows the env's internal sampler to reach far goals.
    # We then filter by our own [min, max] distance criteria.
    vec_env = DummyVecEnv([lambda xml=w["xml_path"]: TimeLimit(
        BunkerEnv(xml_path=xml, render_mode=None,
                  max_goal_sampling_distance=args.max_goal_distance,
                  energy_weight=0.0),
        max_episode_steps=600,
    )])
    raw_env = vec_env.envs[0].env

    episodes = []
    seed = 0
    skipped = 0
    while len(episodes) < args.n_episodes:
        obs, _ = raw_env.reset(seed=seed)
        seed += 1

        start = obs["achieved_goal"]   # [x, y, yaw]
        goal  = obs["desired_goal"]    # [x, y, yaw]
        dist  = float(np.linalg.norm(goal[:2] - start[:2]))

        if dist < args.min_goal_distance or dist > args.max_goal_distance:
            skipped += 1
            continue

        episodes.append({
            "initial_pose":       start.tolist(),
            "goal_pose":          goal.tolist(),
            "straight_line_dist": round(dist, 4),
        })

    vec_env.close()
    del vec_env
    gc.collect()   # release MuJoCo callbacks before next world's env is created

    dists = [ep["straight_line_dist"] for ep in episodes]
    print(f"    Done. Seeds tried: {seed} ({skipped} skipped). "
          f"Dist: min={min(dists):.1f}m, max={max(dists):.1f}m, mean={np.mean(dists):.1f}m")
    print(f"    ep0: start={episodes[0]['initial_pose']}, goal={episodes[0]['goal_pose']}")

    benchmark["worlds"].append({
        "path":       w["path"],
        "difficulty": w["difficulty"],
        "episodes":   episodes,
    })

with open(out_path, "w") as f:
    json.dump(benchmark, f, indent=2)

print(f"\nBenchmark saved: {out_path}")
print(f"  {len(benchmark['worlds'])} worlds × {args.n_episodes} episodes = "
      f"{len(benchmark['worlds']) * args.n_episodes} total")
