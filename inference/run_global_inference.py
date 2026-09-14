#!/usr/bin/env python3
"""Global paired benchmark: OMPL global planner + multiple local controllers.

Runs in three stages, distributed over processes:
  1. plan  - one process per world, plans that world's episodes
  2. exec  - one process per (world, controller), rolls out on those paths
  3. merge - rebuilds episodes.json / summary.json / summary.txt

The OMPL search, the benchmark JSON, the per-episode shared path, the episode
order and the per-world batching are identical to a single-process run. Every
stage writes its own file, so an interrupted run resumes where it stopped.
"""

import os
import sys
import gc
import time
import json
import argparse
import shutil
import subprocess
from collections import Counter
import numpy as np
from scipy import stats
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(__file__, os.pardir, os.pardir)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from envs.gym_bunker_env import BunkerEnv
from gymnasium.wrappers import TimeLimit
from baselines import build_classical_controller, classical_types, is_classical_type
from envs.ompl_global_planner import OMPLGlobalPlanner
from stats_utils import wilcoxon_paired, mcnemar_paired, fmt_p, PAIRED_METRICS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Per-step signals stored in _traces/world_XX__label.npz (float32, flat arrays
# concatenated over the world's episodes, with a parallel episode_id array).
TRACE_FIELDS = ("x", "y", "yaw", "v", "w", "cmd_v", "cmd_w", "energy_j",
                "current_r", "current_l", "cte", "progress_s",
                "target_x", "target_y", "dt_ms")

class PathFollower:
    """
    Arc-length lookahead on a polyline path.

    For each control step the caller provides the current robot position and
    receives a lookahead target that lies exactly on the path polyline:

      1. Project the robot onto the closest segment of the polyline.
      2. Record the arc-length *progress* at that projection.
      3. Advance a fixed *lookahead distance* along the polyline from the
         point of maximum progress so far (monotonic -- never goes backward).
      4. Interpolate the target point and its tangent direction.

    The monotonic-progress guarantee prevents the controller from oscillating
    if the robot momentarily drifts closer to an earlier segment.

    Parameters
    ----------
    waypoints : list of (x, y, yaw) tuples
        Output of OMPLGlobalPlanner.plan_path().
    lookahead_distance : float
        Arc-length advance from progress point (meters).
    """

    def __init__(self, waypoints, lookahead_distance=1.5):
        pts = np.array(waypoints, dtype=np.float64)
        self.waypoints = pts                       # (N, 3) [x, y, yaw]
        self.xy = pts[:, :2].copy()                # (N, 2)
        self.lookahead = float(lookahead_distance)
        self.n_pts = len(pts)

        # Pre-compute segment geometry
        if self.n_pts < 2:
            self.seg_vecs = np.zeros((0, 2))
            self.seg_lengths = np.zeros(0)
            self.seg_len_sq = np.zeros(0)
            self.cum_arc = np.zeros(max(self.n_pts, 1))
            self.total_length = 0.0
        else:
            self.seg_vecs = np.diff(self.xy, axis=0)                      # (N-1, 2)
            self.seg_lengths = np.linalg.norm(self.seg_vecs, axis=1)      # (N-1,)
            self.seg_len_sq = self.seg_lengths ** 2                       # (N-1,)
            self.cum_arc = np.zeros(self.n_pts)
            self.cum_arc[1:] = np.cumsum(self.seg_lengths)
            self.total_length = float(self.cum_arc[-1])

        # Tracking state (reset per episode via reset())
        self._max_progress = 0.0
        self._last_target_xy = None

    # ── public API ──────────────────────────────────────────────────────────

    def reset(self):
        """Reset tracking state for a new episode / controller run."""
        self._max_progress = 0.0
        self._last_target_xy = None

    def get_lookahead_target(self, robot_xy):
        """
        Compute the lookahead target on the path polyline.

        Returns
        -------
        target_xy       : ndarray (2,)  lookahead point on the polyline
        target_yaw      : float         path tangent angle at the target
        progress_s      : float         raw arc-length of closest projection
        cross_track_err : float         distance from robot to closest point
        tangent_angle   : float         path tangent at the closest point
        actual_la_dist  : float         Euclidean robot → target distance
        target_changed  : bool          whether target moved > 1 cm since last call
        """
        robot_xy = np.asarray(robot_xy, dtype=np.float64)[:2]

        # Degenerate: single-point path
        if self.n_pts < 2:
            pt = self.xy[0] if self.n_pts else np.zeros(2)
            yaw = float(self.waypoints[0, 2]) if self.n_pts else 0.0
            d = float(np.linalg.norm(robot_xy - pt))
            changed = self._check_target_changed(pt)
            return pt.copy(), yaw, 0.0, d, yaw, d, changed

        # 1. Closest point on the polyline (raw, for CTE and heading error)
        raw_progress, _, cross_track = self._closest_point_on_polyline(robot_xy)

        # 2. Monotonic progress for target selection
        effective_progress = max(raw_progress, self._max_progress)
        self._max_progress = effective_progress

        # 3. Advance lookahead along the polyline
        target_s = min(effective_progress + self.lookahead, self.total_length)
        target_xy = self._point_at_arc_length(target_s)
        target_yaw = self._tangent_at_arc_length(target_s)

        # 4. Tangent at the raw closest point (for heading-error diagnostics)
        tangent_angle = self._tangent_at_arc_length(raw_progress)

        # 5. Actual Euclidean distance robot → target
        actual_la_dist = float(np.linalg.norm(robot_xy - target_xy))

        # 6. Did the target move since last call?
        changed = self._check_target_changed(target_xy)

        return target_xy, target_yaw, raw_progress, cross_track, tangent_angle, actual_la_dist, changed

    # ── internals ───────────────────────────────────────────────────────────

    def _check_target_changed(self, target_xy, tol=0.01):
        target_xy = np.asarray(target_xy)
        if self._last_target_xy is None:
            self._last_target_xy = target_xy.copy()
            return True
        changed = float(np.linalg.norm(target_xy - self._last_target_xy)) > tol
        self._last_target_xy = target_xy.copy()
        return changed

    def _closest_point_on_polyline(self, point):
        """Return (arc_length, closest_point, distance)."""
        best_dist_sq = np.inf
        best_s = 0.0
        best_pt = self.xy[0].copy()

        for i in range(self.n_pts - 1):
            a = self.xy[i]
            seg = self.seg_vecs[i]
            lsq = self.seg_len_sq[i]

            if lsq < 1e-24:                       # degenerate (repeated point)
                proj = a
                t = 0.0
            else:
                t = float(np.dot(point - a, seg) / lsq)
                t = max(0.0, min(1.0, t))
                proj = a + t * seg

            d_sq = float(np.sum((point - proj) ** 2))
            if d_sq < best_dist_sq:
                best_dist_sq = d_sq
                best_s = self.cum_arc[i] + t * self.seg_lengths[i]
                best_pt = proj

        return float(best_s), best_pt, float(np.sqrt(max(best_dist_sq, 0.0)))

    def _point_at_arc_length(self, s):
        """Interpolate a point on the polyline at arc length *s*."""
        s = max(0.0, min(s, self.total_length))
        if s <= 0.0:
            return self.xy[0].copy()
        if s >= self.total_length:
            return self.xy[-1].copy()

        idx = int(np.searchsorted(self.cum_arc[1:], s, side="left"))
        idx = max(0, min(idx, self.n_pts - 2))

        seg_len = self.seg_lengths[idx]
        if seg_len < 1e-12:
            return self.xy[idx].copy()

        t = (s - self.cum_arc[idx]) / seg_len
        return self.xy[idx] + max(0.0, min(1.0, t)) * self.seg_vecs[idx]

    def _tangent_at_arc_length(self, s):
        """Path tangent angle (radians) at arc length *s*."""
        s = max(0.0, min(s, self.total_length))
        if self.n_pts < 2:
            return 0.0

        if s >= self.total_length:
            idx = self.n_pts - 2
        else:
            idx = int(np.searchsorted(self.cum_arc[1:], s, side="left"))
            idx = max(0, min(idx, self.n_pts - 2))

        seg = self.seg_vecs[idx]
        if np.linalg.norm(seg) < 1e-12:
            # Skip degenerate segments forward
            for j in range(idx + 1, self.n_pts - 1):
                if self.seg_lengths[j] > 1e-12:
                    seg = self.seg_vecs[j]
                    break
            else:
                return 0.0

        return float(np.arctan2(seg[1], seg[0]))


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def batch_ci95(values):
    """Mean and 95 % CI (t-distribution) from a list of scalars."""
    arr = np.array(values, dtype=float)
    n = len(arr)
    if n == 0:
        return None, None
    mean = float(np.mean(arr))
    if n < 2:
        return mean, 0.0
    sd = float(np.std(arr, ddof=1))
    h = float(stats.t.ppf(0.975, n - 1) * sd / np.sqrt(n))
    return mean, h


def polyline_length(waypoints):
    """Total Euclidean length of a polyline [(x,y,...), ...]."""
    pts = np.array(waypoints)[:, :2]
    if len(pts) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def path_geometry_descriptors(waypoints, straight_line_dist):
    """
    Compute geometry descriptors for a planner path polyline.

    Parameters
    ----------
    waypoints : list of (x, y, ...) tuples
    straight_line_dist : float – Euclidean start-to-goal distance

    Returns
    -------
    dict with:
        tortuosity                 – path_length / straight_line_dist  (>=1, 1=straight)
        cumulative_heading_change  – Σ |Δθ_i|  over consecutive segments (rad)
        max_turn_angle             – max |Δθ_i| between consecutive segments (rad)
        mean_curvature_proxy       – cumulative_heading_change / path_length  (rad/m)
    """
    pts = np.array(waypoints, dtype=np.float64)[:, :2]
    n = len(pts)

    if n < 2:
        return {
            "tortuosity": 1.0,
            "cumulative_heading_change": 0.0,
            "max_turn_angle": 0.0,
            "mean_curvature_proxy": 0.0,
        }

    seg_vecs = np.diff(pts, axis=0)                              # (N-1, 2)
    seg_lens = np.linalg.norm(seg_vecs, axis=1)                  # (N-1,)
    path_len = float(np.sum(seg_lens))

    # Tortuosity
    tortuosity = path_len / max(straight_line_dist, 1e-9)

    # Heading of each non-degenerate segment
    valid = seg_lens > 1e-12
    headings = np.arctan2(seg_vecs[valid, 1], seg_vecs[valid, 0])  # (M,)

    if len(headings) < 2:
        return {
            "tortuosity": round(tortuosity, 6),
            "cumulative_heading_change": 0.0,
            "max_turn_angle": 0.0,
            "mean_curvature_proxy": 0.0,
        }

    # Heading changes wrapped to [-pi, pi]
    dh = np.diff(headings)
    dh = np.arctan2(np.sin(dh), np.cos(dh))  # wrap
    abs_dh = np.abs(dh)

    cum_heading = float(np.sum(abs_dh))
    max_turn = float(np.max(abs_dh))
    curvature_proxy = cum_heading / max(path_len, 1e-9)

    return {
        "tortuosity": round(tortuosity, 6),
        "cumulative_heading_change": round(cum_heading, 6),
        "max_turn_angle": round(max_turn, 6),
        "mean_curvature_proxy": round(curvature_proxy, 6),
    }


def angle_diff(a, b):
    """Signed difference a - b wrapped to [-pi, pi]."""
    d = a - b
    return float(np.arctan2(np.sin(d), np.cos(d)))


def parse_controller_spec(spec_str):
    """Parse a controller spec.

    SAC:       'label:SAC'  -> energy_weight/step_weight read from run_config.txt
    Classical: 'label:TYPE:energy_weight' for any registered baseline (e.g. NMPC, DWA)
    """
    parts = spec_str.split(":")
    if len(parts) not in (2, 3):
        raise ValueError(
            f"Controller spec must be 'label:SAC' or 'label:TYPE:energy_weight', got '{spec_str}'"
        )
    label, ctrl_type = parts[0], parts[1].upper()
    valid = ["SAC"] + classical_types()
    if ctrl_type == "SAC":
        if len(parts) == 3:
            raise ValueError(f"SAC energy_weight is read from run_config.txt; use 'label:SAC' (got '{spec_str}')")
        ew = None
        sw = None
    elif is_classical_type(ctrl_type):
        if len(parts) != 3:
            raise ValueError(f"{ctrl_type} spec requires energy_weight: 'label:{ctrl_type}:energy_weight', got '{spec_str}'")
        ew = float(parts[2])
        sw = 1.0  # unused by classical controllers
    else:
        raise ValueError(f"Controller type must be one of {valid}, got '{ctrl_type}'")
    return {
        "label": label,
        "type": ctrl_type,
        "energy_weight": ew,
        "step_weight": sw,
        "run": label if ctrl_type == "SAC" else None,
    }


def _fmt_ci(val):
    """Format (mean, ci) or None for display."""
    if val is None:
        return "N/A"
    mean, ci = val
    if mean is None:
        return "N/A"
    return f"{mean:.4f} +/- {ci:.4f}"


def _fmt_pct(val):
    if val is None:
        return "N/A"
    return f"{val * 100:.1f}%"


def _fmt_ci_pct(val):
    """Format (mean, ci) tuple as percentage with CI."""
    if val is None:
        return "N/A"
    mean, ci = val
    if mean is None:
        return "N/A"
    return f"{mean * 100:.1f}% +/- {ci * 100:.1f}%"


# ═══════════════════════════════════════════════════════════════════════════════
# Episode runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_controller_episode(
    vec_env, raw_env, controller, ctrl_info, path_follower,
    initial_pose, final_goal, max_ep_len, dt_sim, goal_threshold,
):
    """
    Execute a single controller on an already-planned path.

    Returns a dict with all per-episode metrics.
    """
    # Reset env to the exact initial state
    vec_env.reset()
    obs_list = vec_env.env_method("set_manual_pose", initial_pose, final_goal)
    obs = {k: np.array([v]) for k, v in obs_list[0].items()}

    is_deterministic = ctrl_info["deterministic"]
    is_sac = ctrl_info["type"] == "SAC"

    # Reset per-episode controller state (classical warm-starts/filters)
    if not is_sac:
        controller.reset()

    # Reset path follower tracking state
    path_follower.reset()

    # Accumulators
    ep_path_length = 0.0
    ep_energy_sum = 0.0
    ep_dtimes = []
    alphas_sq = []   # (dω/dt)² samples; per-episode RMS = sqrt(mean), rad/s²
    outcome = "TIMEOUT"

    # Per-step trace: everything the loop already computes, kept instead of
    # discarded. Written to _traces/ by stage_exec, never into episodes.json.
    trace = {k: [] for k in TRACE_FIELDS}

    # Seed previous yaw rate with post-reset body-frame angular velocity
    _, w_body_init = raw_env.get_robot_velocities()
    prev_omega = float(w_body_init[2])

    for step in range(max_ep_len):
        # ── Robot state ──
        robot_xy = raw_env.data.qpos[:2].copy()
        qw, _, _, qz = raw_env.data.qpos[3:7]
        robot_yaw = float(2.0 * np.arctan2(qz, qw))

        # Pre-step signals for the trace (pure reads, no state change).
        # The last two observation entries are the normalised motor currents.
        obs_vec = np.asarray(obs["observation"][0] if isinstance(obs, dict) else obs[0])
        cur_r, cur_l = raw_env.unnormalize_current(obs_vec[-2:])
        v_body_t, w_body_t = raw_env.get_robot_velocities()

        # ── Arc-length lookahead ──
        (
            target_xy, target_yaw, progress_s,
            cte, tangent_angle, actual_la, target_changed,
        ) = path_follower.get_lookahead_target(robot_xy)

        # ── Set intermediate goal ──
        goal_3d = np.array(
            [target_xy[0], target_xy[1], target_yaw], dtype=np.float32
        )
        vec_env.env_method("set_inference_goal", goal_3d)

        # ── Predict action ──
        t0 = time.perf_counter()
        if is_sac:
            action, _ = controller.predict(obs, deterministic=is_deterministic)
        else:
            action, _ = controller.predict(obs)
        ep_dtimes.append((time.perf_counter() - t0) * 1000.0)

        # ── Step environment ──
        obs, _, dones, info = vec_env.step(action)
        inf = info[0]

        # ── Track displacement / energy ──
        # Skip path on terminal step: DummyVecEnv auto-resets on done, so
        # qpos reflects the new episode's spawn point, not the final position.
        if not dones[0]:
            robot_xy_new = raw_env.data.qpos[:2].copy()
            ep_path_length += float(np.linalg.norm(robot_xy_new - robot_xy))
        else:
            robot_xy_new = robot_xy  # keep last known position for downstream checks
        step_energy = float(inf.get("energy_joules", 0.0))
        ep_energy_sum += step_energy
        for k, v in (("x", robot_xy[0]), ("y", robot_xy[1]), ("yaw", robot_yaw),
                     ("v", v_body_t[0]), ("w", w_body_t[2]),
                     ("cmd_v", action[0][0]), ("cmd_w", action[0][1]),
                     ("energy_j", step_energy), ("current_r", cur_r), ("current_l", cur_l),
                     ("cte", cte), ("progress_s", progress_s),
                     ("target_x", target_xy[0]), ("target_y", target_xy[1]),
                     ("dt_ms", ep_dtimes[-1])):
            trace[k].append(v)
        if not dones[0]:
            _, curr_w = raw_env.get_robot_velocities()
            omega_t = float(curr_w[2])
            alpha = (omega_t - prev_omega) / dt_sim
            alphas_sq.append(alpha * alpha)
            prev_omega = omega_t
        dist_to_final = float(np.linalg.norm(robot_xy_new - final_goal[:2]))

        if dist_to_final < goal_threshold:
            outcome = "SUCCESS"
            break
        if inf.get("collision", False):
            outcome = "COLLISION"
            break
        if dones[0]:  # TimeLimit truncation (collision already caught above)
            outcome = "TIMEOUT"
            break

    steps_taken = step + 1
    total_len = path_follower.total_length

    return {
        "outcome": outcome,
        "steps": steps_taken,
        "elapsed_time": round(steps_taken * dt_sim, 6),

        # Path & energy
        "executed_path_length": round(ep_path_length, 6),
        "planner_path_length": round(total_len, 6),
        "energy_total": round(ep_energy_sum, 6),
        "energy_per_executed_meter": (
            round(ep_energy_sum / ep_path_length, 6)
            if ep_path_length > 1e-6 else None
        ),

        # Euclidean SPL (filled in by caller with straight_line_dist)
        "spl_euclidean": None,

        # Angular control
        "alpha_rms": round(float(np.sqrt(np.mean(alphas_sq))), 6) if alphas_sq else None,

        # Decision time
        "decision_time_ms_mean": round(float(np.mean(ep_dtimes)), 4),

        # Per-step trace, popped by stage_exec before serialisation
        "_trace": {k: np.asarray(v, dtype=np.float32) for k, v in trace.items()},
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Summary computation
# ═══════════════════════════════════════════════════════════════════════════════

def compute_summary(records, ctrl_specs, config, n_batches_ci=5):
    """Aggregate per-episode records into summary statistics.

    Uses the batch-means method to compute
    95% CIs that correctly account for within-world episode correlation.

    For each world the episodes are split into n_batches_ci equal groups.
    Per-batch rates and metric means are computed, then batch_ci95 is applied
    over all (world × batch) batch-level values.

    Parameters
    ----------
    n_batches_ci : int
        Number of batches per world for the batch-means CI (default 5).
    """
    from collections import defaultdict

    ctrl_labels = [s["label"] for s in ctrl_specs]
    difficulties = sorted(set(r["difficulty"] for r in records))

    summary = {"config": config, "per_difficulty": {}, "overall": {},
               "paired_deltas": {}, "paired_tests": {}}

    # ── Helper: build batch-level values for one group of records ──────────
    def _batch_stats(eps_subset, label, n_total_for_e2e):
        """
        Given a flat list of episode records and a controller label, return
        batch-level aggregates (one value per world-batch) for CI computation.

        Returns a dict keyed by metric name → list of batch-level values.
        An empty list means no data; caller should return (None, None) for CI.
        """
        # Group by world_id
        by_world = defaultdict(list)
        for r in eps_subset:
            by_world[r["world_id"]].append(r)

        # Accumulate one scalar per (world, batch)
        agg = defaultdict(list)

        for world_eps in by_world.values():
            batches = np.array_split(world_eps, n_batches_ci)
            for batch in batches:
                if len(batch) == 0:
                    continue

                # Collect controller results for this batch
                ctrl_in_batch = []
                for r in batch:
                    cr = r["controller_results"].get(label)
                    if cr and cr.get("outcome") != "PLAN_FAIL":
                        ctrl_in_batch.append((r, cr))

                succ_in_batch = [(r, cr) for r, cr in ctrl_in_batch
                                 if cr["outcome"] == "SUCCESS"]
                n_b = len(ctrl_in_batch)
                n_s_b = len(succ_in_batch)

                if n_b == 0:
                    continue

                # Rate metrics (controller-conditioned)
                agg["sr_given_plan"].append(n_s_b / n_b)
                agg["cr_given_plan"].append(
                    sum(1 for _, cr in ctrl_in_batch if cr["outcome"] == "COLLISION") / n_b
                )
                agg["tr_given_plan"].append(
                    sum(1 for _, cr in ctrl_in_batch if cr["outcome"] == "TIMEOUT") / n_b
                )

                # All-episode continuous metrics
                dt_vals = [cr["decision_time_ms_mean"] for _, cr in ctrl_in_batch
                           if cr.get("decision_time_ms_mean") is not None]
                if dt_vals:
                    agg["decision_time_ms"].append(float(np.mean(dt_vals)))

                # Success-only continuous metrics
                if n_s_b == 0:
                    continue

                def _mean_succ(key):
                    vals = [cr[key] for _, cr in succ_in_batch if cr.get(key) is not None]
                    return float(np.mean(vals)) if vals else None

                # Map per-episode record keys → summary agg keys
                for ep_key, agg_key in [
                    ("energy_total",             "eavg"),
                    ("energy_per_executed_meter", "jpm"),
                    ("spl_euclidean",             "spl"),
                    ("steps",                    "steps"),
                    ("alpha_rms",                "alpha_rms"),
                ]:
                    v = _mean_succ(ep_key)
                    if v is not None:
                        agg[agg_key].append(v)

        return agg

    for diff in difficulties + ["all"]:
        eps = records if diff == "all" else [r for r in records if r["difficulty"] == diff]
        n_total = len(eps)
        if n_total == 0:
            continue

        n_plan_ok = sum(1 for r in eps if r["planning_success"])
        planned_eps = [r for r in eps if r["planning_success"]]

        diff_summary = {
            "n_episodes": n_total,
            "n_plan_success": n_plan_ok,
            "planning_success_rate": round(n_plan_ok / n_total, 6) if n_total else 0,
            "plan_fail_rate": round(1 - n_plan_ok / n_total, 6) if n_total else 0,
            "controllers": {},
        }

        for label in ctrl_labels:
            # Flat counts (used for display totals only)
            ctrl_results = []
            for r in planned_eps:
                cr = r["controller_results"].get(label)
                if cr and cr.get("outcome") != "PLAN_FAIL":
                    ctrl_results.append(cr)

            n_ctrl = len(ctrl_results)
            if n_ctrl == 0:
                diff_summary["controllers"][label] = {"n_episodes": 0}
                continue

            n_s = sum(1 for c in ctrl_results if c["outcome"] == "SUCCESS")
            n_c = sum(1 for c in ctrl_results if c["outcome"] == "COLLISION")
            n_t = sum(1 for c in ctrl_results if c["outcome"] == "TIMEOUT")

            # Batch-means aggregates
            agg = _batch_stats(planned_eps, label, n_total)

            def _ci(key):
                vals = agg.get(key, [])
                return batch_ci95(vals) if vals else (None, None)

            diff_summary["controllers"][label] = {
                "n_episodes": n_ctrl,
                "n_success": n_s,
                "n_collision": n_c,
                "n_timeout": n_t,
                "n_batches_ci": n_batches_ci,

                # A. Full system (point estimates only — no CI, denominator changes)
                "end_to_end_success_rate": round(n_s / n_total, 6),
                "end_to_end_collision_rate": round(n_c / n_total, 6),
                "end_to_end_timeout_rate": round(n_t / n_total, 6),

                # B. Controller-conditioned rates with batch-means CI
                "controller_success_rate_given_plan": _ci("sr_given_plan"),
                "controller_collision_rate_given_plan": _ci("cr_given_plan"),
                "controller_timeout_rate_given_plan": _ci("tr_given_plan"),

                # C. Success-only aggregates with batch-means CI
                "eavg":      _ci("eavg"),
                "jpm":       _ci("jpm"),
                "spl":       _ci("spl"),
                "steps":     _ci("steps"),
                "alpha_rms": _ci("alpha_rms"),

                # D. All-episode aggregates with batch-means CI
                "decision_time_ms": _ci("decision_time_ms"),
            }

        if diff == "all":
            summary["overall"] = diff_summary
        else:
            summary["per_difficulty"][diff] = diff_summary

    # ── Paired deltas ──────────────────────────────────────────────────────
    if len(ctrl_labels) >= 2:
        for diff in difficulties + ["all"]:
            planned = (
                [r for r in records if r["planning_success"]]
                if diff == "all"
                else [r for r in records if r["difficulty"] == diff and r["planning_success"]]
            )

            paired = {}
            for i, la in enumerate(ctrl_labels):
                for lb in ctrl_labels[i + 1:]:
                    pair_key = f"{la}_vs_{lb}"

                    # Group by world for batch-means on deltas
                    by_world = defaultdict(list)
                    for r in planned:
                        ca = r["controller_results"].get(la, {})
                        cb = r["controller_results"].get(lb, {})
                        if ca.get("outcome") == "SUCCESS" and cb.get("outcome") == "SUCCESS":
                            by_world[r["world_id"]].append((ca, cb))

                    n_paired = sum(len(v) for v in by_world.values())
                    if n_paired == 0:
                        paired[pair_key] = {"n_paired": 0}
                        continue

                    # Per-episode record keys
                    delta_keys = (
                        "energy_total",
                        "energy_per_executed_meter",
                        "spl_euclidean",
                        "steps",
                        "alpha_rms",
                    )

                    delta_agg = defaultdict(list)
                    for world_pairs in by_world.values():
                        batches = np.array_split(world_pairs, n_batches_ci)
                        for batch in batches:
                            if len(batch) == 0:
                                continue
                            for k in delta_keys:
                                vals = [
                                    a[k] - b[k]
                                    for a, b in batch
                                    if a.get(k) is not None and b.get(k) is not None
                                ]
                                if vals:
                                    delta_agg[k].append(float(np.mean(vals)))

                    paired[pair_key] = {"n_paired": n_paired}
                    for k in delta_keys:
                        vals = delta_agg.get(k, [])
                        paired[pair_key][f"delta_{k}"] = (
                            batch_ci95(vals) if vals else (None, None)
                        )

            summary["paired_deltas"][diff if diff != "all" else "all"] = paired

    # Paired significance tests (additive; legacy CIs above are unchanged).
    # Map the shared canonical metric names onto this benchmark's record keys.
    global_key = {"energy": "energy_total", "energy_per_m": "energy_per_executed_meter",
                  "spl": "spl_euclidean", "steps": "steps", "alpha_rms": "alpha_rms"}
    if len(ctrl_labels) >= 2:
        for diff in difficulties + ["all"]:
            planned = (
                [r for r in records if r["planning_success"]]
                if diff == "all"
                else [r for r in records if r["difficulty"] == diff and r["planning_success"]]
            )
            tests = {}
            for i, la in enumerate(ctrl_labels):
                for lb in ctrl_labels[i + 1:]:
                    a_succ, b_succ = [], []
                    metric_pairs = defaultdict(lambda: ([], []))
                    for r in planned:
                        ca = r["controller_results"].get(la, {})
                        cb = r["controller_results"].get(lb, {})
                        sa = ca.get("outcome") == "SUCCESS"
                        sb = cb.get("outcome") == "SUCCESS"
                        a_succ.append(sa)
                        b_succ.append(sb)
                        if sa and sb:
                            for name, _ in PAIRED_METRICS:
                                k = global_key[name]
                                if ca.get(k) is not None and cb.get(k) is not None:
                                    metric_pairs[name][0].append(ca[k])
                                    metric_pairs[name][1].append(cb[k])
                    tests[f"{la}_vs_{lb}"] = {
                        "n_plan": len(planned),
                        "sr_mcnemar": mcnemar_paired(a_succ, b_succ),
                        "metrics": {name: wilcoxon_paired(av, bv)
                                    for name, (av, bv) in metric_pairs.items()},
                    }
            summary["paired_tests"][diff if diff != "all" else "all"] = tests

    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# Human-readable summary formatter
# ═══════════════════════════════════════════════════════════════════════════════

def format_summary(summary, config):
    L = []
    L.append(f"\n{'=' * 80}")
    L.append("GLOBAL PAIRED BENCHMARK RESULTS")
    L.append(f"{'=' * 80}")
    L.append(f"Timestamp:              {config['timestamp']}")
    L.append(f"Lookahead distance:     {config['lookahead_distance']} m")
    L.append(f"OMPL interpolation:     {config['ompl_interpolation_steps']} points")
    L.append(f"OMPL solve budget:      {config['ompl_solve_seconds']} s")
    L.append(f"OMPL inflation radius:  {config['ompl_inflation_radius']} m")
    L.append(f"Max episode steps:      {config['max_ep_steps']}")
    L.append(f"Episodes/world:         {config['n_episodes_per_world']}")
    L.append(f"Goal distance range:    {config['goal_distance_range']} m")
    L.append(f"Seed:                   {config['seed']}")
    L.append(f"Controllers:            {[c['label'] for c in config['controllers']]}")
    for c in config["controllers"]:
        det = "True" if c["type"] == "SAC" else "N/A (NMPC)"
        ckpt = config.get(f"checkpoint_{c['label']}", "N/A")
        sw_str = f", step_weight={c['step_weight']}" if c["type"] == "SAC" else ""
        L.append(
            f"  {c['label']}: type={c['type']}, energy_weight={c['energy_weight']}{sw_str}, "
            f"deterministic={det}"
            + (f"\n    checkpoint: {ckpt}" if c["type"] == "SAC" else "")
        )

    sections = list(summary.get("per_difficulty", {}).items())
    if summary.get("overall"):
        sections.append(("ALL", summary["overall"]))

    for diff_name, ds in sections:
        L.append(f"\n{'=' * 80}")
        L.append(f"DIFFICULTY: {diff_name.upper()}")
        L.append(f"{'=' * 80}")
        L.append(
            f"Episodes: {ds['n_episodes']}  |  Planned: {ds['n_plan_success']}  |  "
            f"Plan fail: {_fmt_pct(ds['plan_fail_rate'])}"
        )

        for label, cs in ds.get("controllers", {}).items():
            if cs.get("n_episodes", 0) == 0:
                continue

            L.append(f"\n  {'-' * 74}")
            L.append(f"  Controller: {label}")
            L.append(f"  {'-' * 74}")

            L.append(f"  A. FULL SYSTEM METRICS (all {ds['n_episodes']} episodes)")
            L.append(f"    End-to-end success rate:    {_fmt_pct(cs.get('end_to_end_success_rate'))}")
            L.append(f"    End-to-end collision rate:  {_fmt_pct(cs.get('end_to_end_collision_rate'))}")
            L.append(f"    End-to-end timeout rate:    {_fmt_pct(cs.get('end_to_end_timeout_rate'))}")
            L.append(f"    Plan fail rate:             {_fmt_pct(ds.get('plan_fail_rate'))}")

            n_batches_ci = cs.get("n_batches_ci", "?")
            L.append(
                f"  B. CONTROLLER-CONDITIONED ({cs['n_episodes']} planned episodes, "
                f"batch-means CI: {n_batches_ci} batches/world)"
            )
            L.append(
                f"    Success Rate (|plan):   {_fmt_ci_pct(cs.get('controller_success_rate_given_plan'))}  "
                f"({cs['n_success']}/{cs['n_episodes']})"
            )
            L.append(
                f"    Collision Rate (|plan): {_fmt_ci_pct(cs.get('controller_collision_rate_given_plan'))}  "
                f"({cs['n_collision']}/{cs['n_episodes']})"
            )
            L.append(
                f"    Timeout Rate (|plan):   {_fmt_ci_pct(cs.get('controller_timeout_rate_given_plan'))}  "
                f"({cs['n_timeout']}/{cs['n_episodes']})"
            )

            L.append(f"  C. SUCCESS-ONLY METRICS ({cs['n_success']} episodes)")
            L.append(f"    SPL:               {_fmt_ci(cs.get('spl'))}")
            L.append(f"    Eavg (success):    {_fmt_ci(cs.get('eavg'))} J")
            L.append(f"    J/m  (success):    {_fmt_ci(cs.get('jpm'))} J/m")
            L.append(f"    RMS ang.acc (succ):{_fmt_ci(cs.get('alpha_rms'))} rad/s^2")
            L.append(f"    Steps (success):   {_fmt_ci(cs.get('steps'))}")

            L.append(f"  D. ALL-EPISODE METRICS")
            L.append(f"    Decision time:     {_fmt_ci(cs.get('decision_time_ms'))} ms/step")

    pt = summary.get("paired_tests", {})
    if pt:
        L.append(f"\n{'=' * 80}")
        L.append("PAIRED SIGNIFICANCE TESTS (additive to the CIs above)")
        L.append("  Wilcoxon signed-rank on per-episode differences (mutual-success episodes);")
        L.append("  McNemar (exact) on paired success/failure. median d is (first - second).")
        L.append(f"{'=' * 80}")
        diff_order = list(summary.get("per_difficulty", {}).keys())
        if summary.get("overall"):
            diff_order.append("all")
        for diff_name in diff_order:
            d = pt.get(diff_name, {})
            if not d:
                continue
            L.append(f"\n  difficulty: {diff_name.upper()}")
            for pair_key, e in d.items():
                a_lbl, b_lbl = pair_key.split("_vs_", 1)
                mc = e["sr_mcnemar"]
                L.append(f"  {a_lbl} vs {b_lbl}  (n_plan={e['n_plan']})")
                L.append(f"    SR  (McNemar):  p={fmt_p(mc['p_value'])}  "
                         f"[only {a_lbl}: {mc['n_a_only']}, only {b_lbl}: {mc['n_b_only']}]")
                for k, w in e["metrics"].items():
                    if w is None:
                        continue
                    L.append(f"    {k:<26} p={fmt_p(w['p_value'])}  "
                             f"median d={w['median_delta']:+.4g}  n={w['n']}")

    L.append(f"\n{'=' * 80}")
    return "\n".join(L)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def check_stack():
    """Warn when the interpreter cannot load the numpy-2 checkpoints."""
    import numpy
    if int(numpy.__version__.split(".")[0]) < 2:
        print(f"WARNING: numpy {numpy.__version__} from {numpy.__file__}", flush=True)
        print("WARNING: numpy < 2 loads only the two re-saved *_ns checkpoints; "
              "run with PYTHONNOUSERSITE=1 inside the pinned env", flush=True)


def build_parser():
    p = argparse.ArgumentParser(description="Parallel global paired benchmark")
    p.add_argument("--controllers", nargs="+", required=True,
                   help="Controller specs: 'label:SAC' or 'label:TYPE:energy_weight'")
    p.add_argument("--benchmark", type=str, default=None)
    p.add_argument("--n_episodes", type=int, default=None)
    p.add_argument("--max_goal_sampling_distance", type=float, default=20.0)
    p.add_argument("--max_ep_steps", type=int, default=600)
    p.add_argument("--lookahead_distance", type=float, default=2.0)
    p.add_argument("--ompl_solve_seconds", type=float, default=30.0)
    p.add_argument("--ompl_interpolation_steps", type=int, default=200)
    p.add_argument("--ompl_inflation_radius", type=float, default=0.1)
    p.add_argument("--difficulty", type=str, default=None,
                   choices=["easy", "medium", "hard"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ci_n_batches", type=int, default=5)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--plan_workers", type=int, default=3,
                   help="Concurrent planning processes (RRT* has a wall-clock budget: "
                        "more workers means fewer samples per path)")
    p.add_argument("--exec_workers", type=int, default=6,
                   help="Concurrent rollout processes")
    p.add_argument("--fresh", action="store_true",
                   help="Ignore finished task files and redo everything")
    p.add_argument("--stage", default="all", choices=["all", "plan", "exec", "merge"])
    p.add_argument("--world_idx", type=int, default=None, help="internal")
    p.add_argument("--ctrl_spec", type=str, default=None, help="internal")
    p.add_argument("--ctrl_label", type=str, default=None, help="internal")
    return p


def resolve_specs(controllers):
    """Parse specs and apply the same label disambiguation as the sequential runner."""
    specs = [parse_controller_spec(s) for s in controllers]
    for s in specs:
        if s["type"] == "SAC":
            cfg_path = os.path.join(ROOT, "training", "log", s["run"], "run_config.txt")
            if not os.path.exists(cfg_path):
                raise FileNotFoundError(f"run_config.txt not found: {cfg_path}")
            cfg = {}
            with open(cfg_path) as f:
                for line in f:
                    if ":" in line:
                        k, v = line.split(":", 1)
                        cfg[k.strip()] = v.strip()
            s["energy_weight"] = float(cfg["energy_weight"])
            s["step_weight"] = float(cfg["step_weight"])
    counts = Counter(s["label"] for s in specs)
    for s in specs:
        if counts[s["label"]] > 1:
            s["label"] = f"{s['label']}_we{s['energy_weight']:g}"
    labels = [s["label"] for s in specs]
    if len(set(labels)) != len(labels):
        raise ValueError("Controller labels must be unique")
    return specs


def load_benchmark(args):
    path = os.path.abspath(
        args.benchmark or os.path.join(ROOT, "inference", "global_benchmark.json"))
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Benchmark not found: {path}\n"
            f"Generate with: python inference/generate_global_benchmark.py")
    with open(path) as f:
        bench = json.load(f)
    eps_per_world = args.n_episodes or bench["n_episodes_per_world"]
    eps_per_world = min(eps_per_world, bench["n_episodes_per_world"])
    worlds = bench["worlds"]
    if args.difficulty is not None:
        worlds = [w for w in worlds if w["difficulty"] == args.difficulty]
        if not worlds:
            raise ValueError(f"No worlds for difficulty '{args.difficulty}'")
    return path, bench, worlds, eps_per_world


def make_env(xml, args):
    """Build the same single-env DummyVecEnv the sequential runner uses."""
    vec_env = DummyVecEnv([lambda: TimeLimit(
        BunkerEnv(
            xml_path=xml, render_mode=None,
            max_goal_sampling_distance=args.max_goal_sampling_distance,
            energy_weight=0.0, inference_mode=True,
        ),
        max_episode_steps=args.max_ep_steps,
    )])
    return vec_env, vec_env.envs[0].env


def enter_scratch(out_dir, name):
    """Give each worker its own CWD so CasADi JIT output cannot collide or escape."""
    d = os.path.join(out_dir, "_scratch", name)
    os.makedirs(d, exist_ok=True)
    os.chdir(d)
    return d


def leave_scratch(d):
    os.chdir(ROOT)
    shutil.rmtree(d, ignore_errors=True)


def paths_file(out_dir, widx):
    return os.path.join(out_dir, "_paths", f"world_{widx:02d}.json")


def ctrl_file(out_dir, widx, label):
    return os.path.join(out_dir, "_ctrl", f"world_{widx:02d}__{label}.json")


def trace_file(out_dir, widx, label):
    return os.path.join(out_dir, "_traces", f"world_{widx:02d}__{label}.npz")


def save_traces(out_dir, widx, label, traces):
    """Write one world-controller's per-step signals as flat float32 arrays.

    Each field is every episode's samples concatenated in episode order, with
    `episode_id` giving the episode of each sample, so a ragged set of episodes
    round-trips without padding:

        z = np.load(path)
        m = z["episode_id"] == eid
        power = z["energy_j"][m] / dt_sim
    """
    path = trace_file(out_dir, widx, label)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    order = list(traces)
    if order:
        flat = {k: np.concatenate([traces[e][k] for e in order]) for k in TRACE_FIELDS}
        eids = np.concatenate([np.full(len(traces[e]["x"]), int(e), np.int32)
                               for e in order])
    else:
        flat = {k: np.zeros(0, np.float32) for k in TRACE_FIELDS}
        eids = np.zeros(0, np.int32)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, episode_id=eids, **flat)
    os.replace(tmp, path)


def stage_plan(args, worlds, eps_per_world, out_dir):
    """Plan every episode of one world, exactly as the sequential runner does."""
    np.random.seed(args.seed)
    widx = args.world_idx
    world = worlds[widx]
    scratch = enter_scratch(out_dir, f"plan_w{widx:02d}")
    xml = os.path.join(ROOT, "assets", "worlds", f"{world['path']}.xml")
    vec_env, raw_env = make_env(xml, args)
    bounds = dict(
        xmin=float(raw_env.xy_min[0]), xmax=float(raw_env.xy_max[0]),
        ymin=float(raw_env.xy_min[1]), ymax=float(raw_env.xy_max[1]),
    )
    planner = OMPLGlobalPlanner(raw_env.model, raw_env.data, bounds)
    meta_base = {
        "solve_seconds": args.ompl_solve_seconds,
        "inflation_radius": args.ompl_inflation_radius,
        "interpolation_steps": args.ompl_interpolation_steps,
    }
    records = []
    for ep_local, ep in enumerate(world["episodes"][:eps_per_world]):
        ip = np.array(ep["initial_pose"], dtype=np.float32)
        goal = np.array(ep["goal_pose"], dtype=np.float32)
        straight = float(ep["straight_line_dist"])
        rec = {
            "episode_id": widx * eps_per_world + ep_local,
            "world_id": world["path"],
            "difficulty": world["difficulty"],
            "start_pose": ep["initial_pose"],
            "goal_pose": ep["goal_pose"],
            "straight_line_dist": straight,
        }
        vec_env.reset()
        vec_env.env_method("set_manual_pose", ip, goal)
        t0 = time.perf_counter()
        wps = planner.plan_path(
            tuple(float(x) for x in ip), tuple(float(x) for x in goal),
            interpolation_steps=args.ompl_interpolation_steps,
            inflation_radius=args.ompl_inflation_radius,
            solve_seconds=args.ompl_solve_seconds,
        )
        plan_ms = (time.perf_counter() - t0) * 1000.0
        if wps is None:
            rec["planning_success"] = False
            rec["planner_time_ms"] = round(plan_ms, 2)
            rec["planner_path"] = None
            rec["planner_path_length"] = None
            rec["planner_metadata"] = dict(meta_base)
        else:
            path_len = polyline_length(wps)
            rec["planning_success"] = True
            rec["planner_path"] = [list(wp) for wp in wps]
            rec["planner_path_length"] = round(path_len, 6)
            rec["planner_time_ms"] = round(plan_ms, 2)
            rec["planner_path_geometry"] = path_geometry_descriptors(wps, straight)
            rec["planner_metadata"] = dict(meta_base, n_waypoints=len(wps))
        records.append(rec)
        print(f"[plan w{widx}] {ep_local + 1}/{eps_per_world} "
              f"{'OK' if wps is not None else 'PLAN_FAIL'} ({plan_ms:.0f} ms)", flush=True)
    tmp = paths_file(out_dir, widx) + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"world_idx": widx, "episodes": records}, f)
    os.replace(tmp, paths_file(out_dir, widx))
    vec_env.close()
    del planner, vec_env, raw_env
    gc.collect()
    leave_scratch(scratch)


def stage_exec(args, worlds, eps_per_world, out_dir):
    """Run one controller over one world's already-planned paths."""
    import torch
    torch.set_num_threads(1)
    np.random.seed(args.seed)
    widx = args.world_idx
    world = worlds[widx]
    spec = resolve_specs([args.ctrl_spec])[0]
    spec["label"] = args.ctrl_label
    scratch = enter_scratch(out_dir, f"exec_w{widx:02d}_{spec['label']}")
    with open(paths_file(out_dir, widx)) as f:
        planned = json.load(f)["episodes"]
    xml = os.path.join(ROOT, "assets", "worlds", f"{world['path']}.xml")
    vec_env, raw_env = make_env(xml, args)
    dt_sim = raw_env.model.opt.timestep * raw_env.frame_skip
    goal_threshold = raw_env.goal_xy_distance_threshold
    if spec["type"] == "SAC":
        ckpt = os.path.join(ROOT, "training", "log", spec["run"],
                            "best_model", "best_model.zip")
        model = SAC.load(ckpt, env=vec_env, device="auto")
        ctrl_info = {"model": model, "type": "SAC",
                     "energy_weight": spec["energy_weight"], "deterministic": True}
    else:
        ctrl = build_classical_controller(spec["type"], raw_env, spec["energy_weight"])
        ctrl_info = {"model": ctrl, "type": spec["type"],
                     "energy_weight": spec["energy_weight"],
                     "deterministic": ctrl.deterministic}
    results = {}
    traces = {}
    for rec in planned:
        eid = str(rec["episode_id"])
        if not rec["planning_success"]:
            results[eid] = {"outcome": "PLAN_FAIL"}
            continue
        ip = np.array(rec["start_pose"], dtype=np.float32)
        goal = np.array(rec["goal_pose"], dtype=np.float32)
        straight = float(rec["straight_line_dist"])
        pf = PathFollower([tuple(wp) for wp in rec["planner_path"]],
                            args.lookahead_distance)
        res = run_controller_episode(
            vec_env, raw_env, ctrl_info["model"], ctrl_info, pf,
            ip, goal, args.max_ep_steps, dt_sim, goal_threshold,
        )
        if res["outcome"] == "SUCCESS":
            exec_len = res["executed_path_length"]
            res["spl_euclidean"] = round(straight / max(straight, exec_len), 6)
        else:
            res["spl_euclidean"] = 0.0
        traces[eid] = res.pop("_trace")
        results[eid] = res
        jpm = res["energy_per_executed_meter"]
        jpm_str = f"{jpm:6.1f}" if jpm is not None else "   N/A"
        print(f"[exec w{widx} {spec['label']}] ep {eid} {res['outcome']:>9s} "
              f"| E: {res['energy_total']:7.1f}J | J/m: {jpm_str} "
              f"| Steps: {res['steps']:3d}", flush=True)
    tmp = ctrl_file(out_dir, widx, spec["label"]) + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"world_idx": widx, "label": spec["label"], "results": results}, f)
    os.replace(tmp, ctrl_file(out_dir, widx, spec["label"]))
    save_traces(out_dir, widx, spec["label"], traces)
    # drop the controller before exiting: the CasADi solver removes its JIT
    # artifacts in its destructor, and os._exit would skip that
    vec_env.close()
    ctrl_info["model"] = None
    del ctrl_info, vec_env, raw_env
    gc.collect()
    leave_scratch(scratch)


def stage_merge(args, specs, worlds, eps_per_world, bench, bench_path, out_dir):
    """Rebuild episodes.json in sequential order and run the standard summary."""
    labels = [s["label"] for s in specs]
    ctrl_data = {}
    for widx in range(len(worlds)):
        for label in labels:
            fp = ctrl_file(out_dir, widx, label)
            if not os.path.exists(fp):
                raise FileNotFoundError(f"missing controller output: {fp}")
            with open(fp) as f:
                ctrl_data[(widx, label)] = json.load(f)["results"]
    all_records = []
    for widx in range(len(worlds)):
        with open(paths_file(out_dir, widx)) as f:
            planned = json.load(f)["episodes"]
        for rec in planned:
            eid = str(rec["episode_id"])
            rec["controller_results"] = {
                label: ctrl_data[(widx, label)][eid] for label in labels
            }
            all_records.append(rec)
    bench_config = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "controllers": specs,
        "lookahead_distance": args.lookahead_distance,
        "ompl_interpolation_steps": args.ompl_interpolation_steps,
        "ompl_solve_seconds": args.ompl_solve_seconds,
        "ompl_inflation_radius": args.ompl_inflation_radius,
        "max_ep_steps": args.max_ep_steps,
        "max_goal_sampling_distance": args.max_goal_sampling_distance,
        "n_episodes_per_world": eps_per_world,
        "difficulty_filter": args.difficulty,
        "benchmark_path": bench_path,
        "seed": args.seed,
        "ci_n_batches": args.ci_n_batches,
        "goal_distance_range": [bench["min_goal_distance"], bench["max_goal_distance"]],
        "parallel": {"plan_workers": args.plan_workers, "exec_workers": args.exec_workers},
    }
    for s in specs:
        if s["type"] == "SAC":
            bench_config[f"checkpoint_{s['label']}"] = os.path.join(
                ROOT, "training", "log", s["run"], "best_model", "best_model.zip")
    with open(os.path.join(out_dir, "episodes.json"), "w") as f:
        json.dump({"config": bench_config, "episodes": all_records}, f, indent=2)
    summary = compute_summary(all_records, specs, bench_config,
                                n_batches_ci=args.ci_n_batches)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    txt = format_summary(summary, bench_config)
    with open(os.path.join(out_dir, "summary.txt"), "w") as f:
        f.write(txt)
    print(txt)


def run_pool(tasks, n_workers, log_dir):
    """Run (name, argv) subprocess tasks with at most n_workers in flight."""
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", PYTHONNOUSERSITE="1")
    import psutil
    pending, running, failed = list(tasks), [], []
    total = len(pending)
    done = 0
    while pending or running:
        while pending and len(running) < n_workers:
            # an NMPC worker needs ~1.9 GB; hold back rather than risk the OOM killer
            free_gb = psutil.virtual_memory().available / 2**30
            if running and free_gb < 2.5:
                print(f"  holding launch: {free_gb:.1f} GB free", flush=True)
                break
            name, argv = pending.pop(0)
            log = open(os.path.join(log_dir, f"{name}.log"), "w")
            proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, env=env)
            running.append((name, proc, log))
            print(f"  launched {name} ({len(running)} running, "
                  f"{len(pending)} queued)", flush=True)
        time.sleep(2.0)
        for item in list(running):
            name, proc, log = item
            if proc.poll() is None:
                continue
            running.remove(item)
            log.close()
            done += 1
            status = "ok" if proc.returncode == 0 else f"FAILED rc={proc.returncode}"
            if proc.returncode != 0:
                failed.append(name)
            print(f"  [{done}/{total}] {name} {status}", flush=True)
    return failed


def main():
    args = build_parser().parse_args()
    bench_path, bench, worlds, eps_per_world = load_benchmark(args)
    specs = resolve_specs(args.controllers)

    if args.output_dir:
        out_dir = args.output_dir
    else:
        out_dir = os.path.join(ROOT, "inference", "results", "global_paired",
                               time.strftime("%Y%m%d_%H%M%S"))
    out_dir = os.path.abspath(out_dir)
    for sub in ("", "_paths", "_ctrl", "_traces", "logs", "_scratch"):
        os.makedirs(os.path.join(out_dir, sub), exist_ok=True)

    # worker stages leave via os._exit: tearing down MuJoCo + OMPL together
    # aborts the interpreter with a double free after the work is already saved
    if args.stage in ("plan", "exec"):
        if args.stage == "plan":
            stage_plan(args, worlds, eps_per_world, out_dir)
        else:
            stage_exec(args, worlds, eps_per_world, out_dir)
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    if args.stage == "merge":
        return stage_merge(args, specs, worlds, eps_per_world, bench, bench_path, out_dir)

    py = sys.executable
    base = [py, os.path.abspath(__file__),
            "--controllers", *args.controllers,
            "--benchmark", bench_path,
            "--n_episodes", str(eps_per_world),
            "--max_goal_sampling_distance", str(args.max_goal_sampling_distance),
            "--max_ep_steps", str(args.max_ep_steps),
            "--lookahead_distance", str(args.lookahead_distance),
            "--ompl_solve_seconds", str(args.ompl_solve_seconds),
            "--ompl_interpolation_steps", str(args.ompl_interpolation_steps),
            "--ompl_inflation_radius", str(args.ompl_inflation_radius),
            "--seed", str(args.seed),
            "--ci_n_batches", str(args.ci_n_batches),
            "--output_dir", out_dir]
    if args.difficulty:
        base += ["--difficulty", args.difficulty]

    check_stack()
    print(f"Output: {out_dir}")
    print(f"Worlds: {len(worlds)}  Episodes/world: {eps_per_world}  "
          f"Controllers: {len(specs)}")
    print(f"Plan workers: {args.plan_workers}  Exec workers: {args.exec_workers}")

    plan_tasks = []
    for widx in range(len(worlds)):
        if not args.fresh and os.path.exists(paths_file(out_dir, widx)):
            continue
        plan_tasks.append((f"plan_w{widx:02d}",
                           base + ["--stage", "plan", "--world_idx", str(widx)]))
    print(f"\n=== Stage 1: planning ({len(plan_tasks)} tasks) ===", flush=True)
    failed = run_pool(plan_tasks, args.plan_workers, os.path.join(out_dir, "logs"))
    if failed:
        sys.exit(f"planning failed: {failed}")

    exec_tasks = []
    for widx in range(len(worlds)):
        for s, raw_spec in zip(specs, args.controllers):
            if not args.fresh and os.path.exists(ctrl_file(out_dir, widx, s["label"])):
                continue
            exec_tasks.append((f"exec_w{widx:02d}_{s['label']}",
                               base + ["--stage", "exec", "--world_idx", str(widx),
                                       "--ctrl_spec", raw_spec,
                                       "--ctrl_label", s["label"]]))
    print(f"\n=== Stage 2: rollouts ({len(exec_tasks)} tasks) ===", flush=True)
    failed = run_pool(exec_tasks, args.exec_workers, os.path.join(out_dir, "logs"))
    if failed:
        sys.exit(f"rollouts failed: {failed}")

    print("\n=== Stage 3: merge ===", flush=True)
    stage_merge(args, specs, worlds, eps_per_world, bench, bench_path, out_dir)
    shutil.rmtree(os.path.join(out_dir, "_scratch"), ignore_errors=True)
    print(f"\nResults: {out_dir}")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
