import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from baselines.base_controller import BaseController


class EADWA(BaseController):
    """Energy-aware Dynamic Window Approach (Fox et al. 1997) with an added energy term.
    Samples the dynamic window, rolls each (v, w) out at constant velocity, and minimizes
    to_goal + obstacle + speed costs plus w_e * predicted energy. Reuses the sim power model."""

    # Gains tuned energy-blind (w_e=0).
    def __init__(self, env, w_e=0.0, predict_time=2.0, n_v=7, n_w=21,
                 to_goal_gain=1.0, obstacle_gain=1.0, speed_gain=0.1,
                 n_nearest_lidar=20, robot_radius=0.4, lidar_gate=-0.2):
        self.env = env
        self.w_e = float(w_e)
        self.predict_time = float(predict_time)
        self.n_v, self.n_w = int(n_v), int(n_w)
        self.to_goal_gain = to_goal_gain     # alpha (heading)
        self.obstacle_gain = obstacle_gain   # beta  (clearance)
        self.speed_gain = speed_gain         # gamma (velocity)
        self.n_nearest_lidar = int(n_nearest_lidar)
        self.robot_radius = float(robot_radius)
        self.lidar_gate = float(lidar_gate)

        self.dt = env.model.opt.timestep * env.frame_skip   # control period
        self.v_max, self.w_max = env.v_max, env.w_max
        self.v_min, self.w_min = -env.v_max, -env.w_max
        self.v_dot = float(env.velocity_controller.a_max)        # linear accel limit (V_d)
        self.w_dot = float(env.velocity_controller.alpha_max)    # angular accel (unbounded -> full range)

        self.n_lidar = env.n_lidar
        self.lidar_max_range = env.lidar_max_range
        self.power = env.velocity_controller.predict_power        # shared energy model

    def _obstacle_points(self, raw_obs, pose):
        lidar = raw_obs[:self.n_lidar * 3].reshape(self.n_lidar, 3)
        d_norm = lidar[:, 2]
        dists = (d_norm + 1.0) / 2.0 * self.lidar_max_range
        idx = np.argsort(dists)[:self.n_nearest_lidar]
        x, y, th = pose
        pts = []
        for i in idx:
            sin_a, cos_a, dn = lidar[i]
            if dn < self.lidar_gate:
                d = dists[i]
                lx = x + d * (cos_a * np.cos(th) - sin_a * np.sin(th))
                ly = y + d * (cos_a * np.sin(th) + sin_a * np.cos(th))
                pts.append((lx, ly))
        return np.array(pts, dtype=float) if pts else np.zeros((0, 2))

    def _rollout(self, pose, v, w):
        """Constant-velocity trajectory over predict_time (yaw updated first, Fox convention)."""
        x, y, th = pose
        n = max(1, int(round(self.predict_time / self.dt)))
        traj = np.empty((n, 2))
        for k in range(n):
            th += w * self.dt
            x += v * np.cos(th) * self.dt
            y += v * np.sin(th) * self.dt
            traj[k] = (x, y)
        return traj, th

    def predict(self, obs_dict):
        batched = obs_dict['observation'].ndim > 1
        raw_obs = obs_dict['observation'][0] if batched else obs_dict['observation']
        pose = np.asarray(obs_dict['achieved_goal'][0] if batched else obs_dict['achieved_goal'], dtype=float)
        goal = np.asarray(obs_dict['desired_goal'][0] if batched else obs_dict['desired_goal'], dtype=float)[:2]

        v_a = float(raw_obs[self.n_lidar * 3]) * self.v_max
        w_a = float(raw_obs[self.n_lidar * 3 + 1]) * self.w_max
        obstacles = self._obstacle_points(raw_obs, pose)

        # Dynamic window V_s ∩ V_d
        v_lo = max(self.v_min, v_a - self.v_dot * self.dt)
        v_hi = min(self.v_max, v_a + self.v_dot * self.dt)
        w_lo = max(self.w_min, w_a - self.w_dot * self.dt)
        w_hi = min(self.w_max, w_a + self.w_dot * self.dt)
        vs = np.linspace(v_lo, v_hi, self.n_v)
        ws = np.linspace(w_lo, w_hi, self.n_w)

        best_cost, best = np.inf, None
        for v in vs:
            for w in ws:
                traj, _ = self._rollout(pose, v, w)

                # obstacle cost: infeasible if the arc passes within robot_radius
                if obstacles.shape[0]:
                    dmin = float(np.linalg.norm(traj[:, None, :] - obstacles[None, :, :], axis=2).min())
                    if dmin <= self.robot_radius:
                        continue
                    obstacle_cost = 1.0 / dmin
                else:
                    obstacle_cost = 0.0

                # goal cost: distance from the trajectory endpoint to the goal
                # (handles forward and reverse motion fairly, unlike a yaw-based heading cost)
                to_goal_cost = float(np.linalg.norm(traj[-1] - goal))

                speed_cost = self.v_max - abs(v)             # prefer fast motion, either direction
                energy = self.power(v, w) * self.dt

                cost = (self.to_goal_gain * to_goal_cost
                        + self.obstacle_gain * obstacle_cost
                        + self.speed_gain * speed_cost
                        + self.w_e * energy)
                if cost < best_cost:
                    best_cost, best = cost, (v, w)

        if best is None:
            return self._fallback(pose, goal, batched)
        return self._action(best[0], best[1], batched)

    def _fallback(self, pose, goal, batched):
        """Surrounded: rotate in place toward the goal."""
        goal_dir = np.arctan2(goal[1] - pose[1], goal[0] - pose[0])
        err = np.arctan2(np.sin(goal_dir - pose[2]), np.cos(goal_dir - pose[2]))
        return self._action(0.0, float(np.clip(err / self.dt, self.w_min, self.w_max)), batched)

    def _action(self, v, w, batched):
        action = np.array([v / self.v_max, w / self.w_max], dtype=float)
        return (np.expand_dims(action, 0), None) if batched else (action, None)
