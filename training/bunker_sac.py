#!/usr/bin/env python3
"""
Training entry-point for the Bunker agent.

Fixed-pool training: you pick which difficulty levels to include via
--difficulties and the agent trains on that combined pool for the whole run.
"""

import argparse
import math
import os
import subprocess
import sys

import gymnasium as gym
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from bunker_callback import BunkerCallback, SmoothedSaveCallback

sys.path.insert(0, os.path.abspath(os.path.join(__file__, os.pardir, os.pardir)))
from envs.gym_bunker_env import BunkerEnv
from feature_extractors.feature_extractor import FeatureExtractor

def make_single_env(xml_path: str, max_ep_steps: int,
                    energy_weight: float, step_weight: float = 1.0):
    """Each worker is pinned to one world for its entire lifetime."""
    def _init():
        env = BunkerEnv(xml_path=xml_path, render_mode=None, n_lidar=449,
                        energy_weight=energy_weight, step_weight=step_weight)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_ep_steps)
        return env
    return _init

def build_pools_from_metadata(worlds_dir: str) -> dict[str, list[str]]:
    """
    Scans `worlds_dir` for metadata JSON files, groups XML paths by difficulty,
    and returns {'easy': [...], 'medium': [...], 'hard': [...]} sorted for reproducibility.
    """
    import json
    groups: dict[str, list[str]] = {"easy": [], "medium": [], "hard": []}

    for fname in sorted(os.listdir(worlds_dir)):
        if not fname.endswith("_metadata.json"):
            continue
        meta_path = os.path.join(worlds_dir, fname)
        with open(meta_path) as f:
            meta = json.load(f)
        difficulty = meta.get("difficulty", "").lower()
        if difficulty not in groups:
            continue
        xml_name = fname.replace("_metadata.json", ".xml")
        xml_path = os.path.join(worlds_dir, xml_name)
        if os.path.exists(xml_path):
            groups[difficulty].append(xml_path)

    return groups


def build_run_name(energy_weight: float, step_weight: float, seed: int) -> str:
    tag = "vanilla_drl" if energy_weight == 0.0 else f"lean_we{energy_weight:g}"
    name = f"{tag}_s{seed}"
    if step_weight != 1.0:
        name += f"_sw{step_weight:g}"
    return name


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    except Exception:
        return "unknown"

def build_model(env, n_envs: int, max_ep_steps: int, log_dir: str, seed: int):
    policy_kwargs = dict(
        features_extractor_class=FeatureExtractor,
        features_extractor_kwargs=dict(
            features_dim=256,
            n_lidar=449,
            max_distance_diagonal=env.get_attr("max_distance_diagonal")[0],
        ),
        net_arch=[512, 512],
    )
    return SAC(
        policy="MultiInputPolicy",
        env=env,
        policy_kwargs=policy_kwargs,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=4,
            goal_selection_strategy="future",
            copy_info_dict=True,
        ),
        batch_size=512,
        learning_rate=3e-4,
        learning_starts=n_envs * max_ep_steps,
        gamma=0.99,
        tau=0.005,
        buffer_size=1_000_000,
        train_freq=(1, "step"),
        gradient_steps=1,
        target_update_interval=2,
        tensorboard_log=os.path.join(log_dir, "tb"),
        seed=seed,
        verbose=1,
        device="cuda",
    )

def train_normal(train_pool: list[str], val_pool: list[str],
                 total_steps: int, max_ep_steps: int,
                 log_dir: str, difficulties: list[str], seed: int,
                 energy_weight: float, step_weight: float = 1.0):
    if os.path.exists(log_dir):
        raise SystemExit(f"Run directory already exists, refusing to overwrite: {log_dir}")
    os.makedirs(log_dir)

    # Smallest multiple of pool size >= 8: keeps workers balanced across worlds
    n_envs = math.ceil(8 / len(train_pool)) * len(train_pool)

    # Vectorised env
    env = SubprocVecEnv([make_single_env(train_pool[i % len(train_pool)], max_ep_steps,
                                         energy_weight, step_weight)
                         for i in range(n_envs)])
    env = VecMonitor(env, filename=os.path.join(log_dir, "monitor"))

    # Evaluation env — one pinned worker per val world
    eval_env = SubprocVecEnv([make_single_env(val_pool[i], max_ep_steps,
                                              energy_weight, step_weight)
                               for i in range(len(val_pool))])
    eval_env = VecMonitor(eval_env)
    n_eval_episodes = len(val_pool) * 5

    # callbacks ────────────────────────────────────────────────────────────────
    smoothed_save_cb = SmoothedSaveCallback(
        model_save_path=os.path.join(log_dir, "best_model"),
        ema_alpha=0.6,   # matches TensorBoard default smoothing
        log_dir=log_dir,
    )
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=None,   # SmoothedSaveCallback handles saving
        log_path=log_dir,
        eval_freq=5_000,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        render=False,
        verbose=1,
        callback_after_eval=smoothed_save_cb,
    )

    callbacks = CallbackList([eval_callback, BunkerCallback()])

    # policy/network definition
    model = build_model(env, n_envs, max_ep_steps, log_dir, seed)

    _write_run_config(log_dir, mode="normal",
                      difficulties=difficulties,
                      train_pool=train_pool, val_pool=val_pool,
                      n_envs=n_envs, max_ep_steps=max_ep_steps,
                      total_steps=total_steps, seed=seed,
                      energy_weight=energy_weight, step_weight=step_weight)

    print(f"Starting normal training on {n_envs} envs | {total_steps} steps")
    print(f"Difficulties: {difficulties} | Train worlds: {len(train_pool)} | Val worlds: {len(val_pool)}")
    print(f"Logging to {log_dir}")

    model.learn(total_timesteps=total_steps, callback=callbacks,
                tb_log_name="run", progress_bar=True)


def _write_run_config(log_dir: str, mode: str, **kwargs):
    """Writes a run_config.txt summarising the hyperparameters for this run."""
    path = os.path.join(log_dir, "run_config.txt")
    from datetime import datetime
    lines = [
        f"Run config — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"mode: {mode}",
    ]
    lines += [
        f"difficulties: {kwargs.get('difficulties', '?')}",
        f"train_worlds: {len(kwargs.get('train_pool', []))}",
        f"val_worlds:   {len(kwargs.get('val_pool', []))}",
    ]
    lines += [
        f"n_envs: {kwargs['n_envs']}",
        f"max_ep_steps: {kwargs['max_ep_steps']}",
        f"total_steps: {kwargs['total_steps']}",
        f"energy_weight: {kwargs['energy_weight']}",
        f"step_weight: {kwargs['step_weight']}",
        f"seed: {kwargs['seed']}",
        f"git_commit: {git_commit()}",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")

def parse_args():
    parser = argparse.ArgumentParser(description="Train the Bunker SAC agent.")
    parser.add_argument(
        "--difficulties", nargs="+",
        choices=["easy", "medium", "hard"],
        default=["easy", "medium", "hard"],
        help="Which difficulty levels to include in the fixed pool. "
             "Determines n_envs: easy→3, easy+medium→6, all→9. "
             "Default: all three.",
    )
    parser.add_argument(
        "--total-steps", type=int, default=5_000_000,
        help="Total training timesteps. Default: 5_000_000.",
    )
    parser.add_argument(
        "--max-ep-steps", type=int, default=600,
        help="Max steps per episode. Default: 600.",
    )
    parser.add_argument(
        "--energy-weight", type=float, required=True,
        help="Energy penalty weight in the reward function.",
    )
    parser.add_argument(
        "--seed", type=int, required=True,
        help="RNG seed for this run. Must differ across runs of the same configuration.",
    )
    parser.add_argument(
        "--step-weight", type=float, default=1.0,
        help="Multiplier on the base -1/step penalty (0=no time penalty, 1=full). Default: 1.0.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    root      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_dir = os.path.join(root, "assets", "worlds", "train")
    val_dir   = os.path.join(root, "assets", "worlds", "val")

    train_groups = build_pools_from_metadata(train_dir)
    val_groups   = build_pools_from_metadata(val_dir)

    base_log_dir = os.path.join(root, "training", "log")
    run_name     = build_run_name(args.energy_weight, args.step_weight, args.seed)
    log_dir      = os.path.join(base_log_dir, run_name)

    # Preserve ordering: easy < medium < hard
    ordered = ["easy", "medium", "hard"]
    selected = [d for d in ordered if d in args.difficulties]

    train_pool = []
    val_pool   = []
    for d in selected:
        train_pool.extend(train_groups[d])
        val_pool.extend(val_groups[d])

    print(f"Normal training | difficulties={selected} | run={run_name}")
    print(f"  Train: {len(train_pool)} worlds | Val: {len(val_pool)} worlds")

    train_normal(
        train_pool=train_pool,
        val_pool=val_pool,
        total_steps=args.total_steps,
        max_ep_steps=args.max_ep_steps,
        log_dir=log_dir,
        difficulties=selected,
        seed=args.seed,
        energy_weight=args.energy_weight,
        step_weight=args.step_weight,
    )
