#!/usr/bin/env python3

import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

import mujoco
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

from envs.gym_bunker_env import BunkerEnv
from inference.run_global_inference import PathFollower, run_controller_episode

CHECKPOINT = os.path.join(REPO_ROOT, "training", "log", "lean_we0.004_s300", "best_model", "best_model.zip")
MAX_EP_STEPS = 600

# Viewer window (px) and initial camera over the frozen path
WINDOW_SIZE = (1280, 720)
CAMERA = {"lookat": [2.6, 1.8, 0.0], "distance": 20.0, "azimuth": 90.0, "elevation": -55.0}

# Load the frozen episode: world, start/goal poses and the pre-planned RRT* path.
demo = json.load(open(os.path.join(os.path.dirname(__file__), "frozen_path.json")))
xml = os.path.join(REPO_ROOT, "assets", "worlds", demo["world_id"] + ".xml")
start = demo["start_pose"]
goal = demo["goal_pose"]

# Build the environment with the MuJoCo viewer.
vec_env = DummyVecEnv([lambda: TimeLimit(
    BunkerEnv(xml_path=xml, render_mode="human", energy_weight=0.0, # energy weight irrelevant for inference
              inference_mode=True, max_goal_sampling_distance=20.0,
              width=WINDOW_SIZE[0], height=WINDOW_SIZE[1], default_camera_config=CAMERA,
              visual_options={mujoco.mjtVisFlag.mjVIS_RANGEFINDER: False}),  # hide LiDAR rays
    max_episode_steps=MAX_EP_STEPS,
)])
raw_env = vec_env.envs[0].env
dt_sim = raw_env.model.opt.timestep * raw_env.frame_skip
goal_threshold = raw_env.goal_xy_distance_threshold

# Load the shipped LEAN-0.004 policy and follow the frozen path.
model = SAC.load(CHECKPOINT, env=vec_env, device="auto")
ctrl_info = {"model": model, "type": "SAC", "energy_weight": 0.004, "deterministic": True}
pf = PathFollower(demo["waypoints"], demo["lookahead_distance"])

try:
    result = run_controller_episode(
        vec_env, raw_env, model, ctrl_info, pf,
        start, goal, MAX_EP_STEPS, dt_sim, goal_threshold,
    )
    # Report the outcome (compare against frozen_path.json's reference values).
    print(f"\nOutcome: {result['outcome']}  |  Energy: {result['energy_total']:.1f} J  |  Steps: {result['steps']}")
finally:
    # Always shut down the env so the MuJoCo/GLFW viewer closes cleanly.
    vec_env.close()
