import json
import numpy as np
import mujoco
from collections import deque

"""
Velocity vector:

data.qvel = [
    qvel[0] 'mobile_base_joint' (m/s) world frame x
    qvel[1] 'mobile_base_joint' (m/s) world frame y
    qvel[2] 'mobile_base_joint' (m/s) world frame z
    qvel[3] 'mobile_base_joint' (rad/s) world frame roll
    qvel[4] 'mobile_base_joint' (rad/s) world frame pitch
    qvel[5] 'mobile_base_joint' (rad/s) world frame yaw
    qvel[6] 'w_rr_joint' (rad/s) speed
    qvel[7] 'w_rc_joint' (rad/s) speed
    qvel[8] 'w_rf_joint' (rad/s) speed
    qvel[9] 'w_lr_joint' (rad/s) speed
    qvel[10] 'w_lc_joint' (rad/s) speed
    qvel[11] 'w_lf_joint' (rad/s) speed
    ]
"""

EPSILON_W = 0.05   # rad/s

class BunkerVelocityController:

    def __init__(self, v0: float = 0.0, w0: float = 0.0, a_max: float = 1.5, alpha_max: float = 1.5,
                 window_size: int = 5,
                 motors_json: str | None = None, calib_json: str | None = None):
        # Motion Control State
        self.v_cmd = self.v_cur = float(v0)
        self.w_cmd = self.w_cur = float(w0)
        self.a_max, self.alpha_max = abs(a_max), np.inf
        self._t_prev: float | None = None

        # Actual MuJoCo body-frame velocities
        self.v_actual: float = 0.0
        self.w_actual: float = 0.0

        # Robot Physical Params (populated by gym_bunker_env after construction)
        self.w_track: float = 0.0
        self.r_wheel: float = 0.0
        self.act_r: list[int] = []
        self.act_l: list[int] = []

        if motors_json is None or calib_json is None:
            raise ValueError("motors_json and calib_json are required")

        with open(motors_json) as f:
            mp = json.load(f)
        self.kt_r       = mp["K_R_sprocket"]
        self.kt_l       = mp["K_L_sprocket"]
        self.tau_c      = mp["tau_c"]
        self.w_eff      = mp["W_eff"]
        self.c_damp     = mp["c"]
        self.b_damp     = mp["b"]
        self.r_sprocket = mp["r_sprocket"]

        with open(calib_json) as f:
            cb = json.load(f)
        self.r_terminal      = cb["R_terminal"]
        self.fidelity_factor = cb["fidelity_factor"]
        self.max_current_real = cb["max_current_real"]
        self.kt_eff_obs      = cb["kt_eff_obs"]
        self.asym_obs        = cb["asym_obs"]
        self.max_current_obs = cb["max_current_obs"]

        # Sliding window for current observation
        self.hist_r = deque(maxlen=window_size)
        self.hist_l = deque(maxlen=window_size)

    def set_cmd(self, v: float, w: float) -> None:
        self.v_cmd, self.w_cmd = float(v), float(w)

    def update_current_estimate(self, torque_r_sum: float, torque_l_sum: float) -> None:
        """Push a new instantaneous torque reading into the sliding window.

        torque_r_sum / torque_l_sum are the sums of actuatorfrc sensor values
        for the three right / left wheel actuators (N·m each).
        kt_eff_obs maps sim actuator force → motor current, fitted from real robot data.
        """
        self.hist_r.append(torque_r_sum / self.kt_eff_obs)
        self.hist_l.append(torque_l_sum / self.kt_eff_obs)

    def get_average_current(self) -> tuple[float, float]:
        """Return smoothed motor currents (A) from the sliding window.

        asym_obs scales the left reading to match the real robot's sensor ratio.
        """
        if not self.hist_r:
            return 0.0, 0.0

        avg_r = sum(self.hist_r) / len(self.hist_r)
        avg_l = sum(self.hist_l) / len(self.hist_l)

        avg_l *= self.asym_obs

        # Quantization, this reproduces the 0.1 A step behaviour of the real
        # AgileX Bunker sensor (the mminmum resolution)
        quantized_r = round(avg_r, 1)
        quantized_l = round(avg_l, 1)

        # Real Bunker convention: negative current = forward
        return -quantized_r, -quantized_l

    # Energy model
    def predict_power(self, v: float, w: float) -> float:
        """Power (Watts) from the identified dynamics model at body velocity (v, w).
        """
        s = self.c_damp * v * self.r_sprocket
        d = (self.b_damp * w + self.tau_c * np.tanh(w / EPSILON_W)) * 2.0 * self.r_sprocket / self.w_eff
        i_r = (s + d) / (2.0 * self.kt_r)
        i_l = (s - d) / (2.0 * self.kt_l)

        omega_spr_R = (v + w * self.w_eff / 2.0) / self.r_sprocket
        omega_spr_L = (v - w * self.w_eff / 2.0) / self.r_sprocket

        p_mech    = self.kt_r * abs(i_r) * abs(omega_spr_R) + self.kt_l * abs(i_l) * abs(omega_spr_L)
        p_heat    = self.r_terminal * (i_r ** 2 + i_l ** 2)
        p_coulomb = self.tau_c * abs(np.tanh(w / EPSILON_W))

        return (p_mech + p_heat + p_coulomb) * self.fidelity_factor

    def get_energy_consumption(self) -> float:
        """Estimate power (Watts) at the actual MuJoCo body-frame velocities."""
        return self.predict_power(self.v_actual, self.w_actual)

    def reset(self) -> None:
        self.v_cmd = self.v_cur = 0.0
        self.w_cmd = self.w_cur = 0.0
        self.v_actual = self.w_actual = 0.0
        self.hist_r.clear()
        self.hist_l.clear()
        self._t_prev = None

    def __call__(self, m: mujoco.MjModel, d: mujoco.MjData) -> None:
        dt = m.opt.timestep if self._t_prev is None else max(d.time - self._t_prev, 1e-9)
        self._t_prev = d.time
        self.v_cur = self._rate_limit(self.v_cur, self.v_cmd, self.a_max,     dt)
        self.w_cur = self._rate_limit(self.w_cur, self.w_cmd, self.alpha_max, dt)

        # Read actual body-frame velocities (needed by energy model)
        vx, vy = d.qvel[0], d.qvel[1]
        w_yaw  = d.qvel[5]
        body_id = m.body('mobile_base').id
        R = d.xmat[body_id].reshape(3, 3)
        self.v_actual = float(R[0, 0] * vx + R[0, 1] * vy)
        self.w_actual = float(w_yaw)

        # Kinematic wheel speed mapping
        w_kinematic = self.w_eff / 0.60
        w_r = (2 * self.v_cur + self.w_cur * w_kinematic) / (2 * self.r_wheel)
        w_l = (2 * self.v_cur - self.w_cur * w_kinematic) / (2 * self.r_wheel)
        for idx in self.act_r: d.ctrl[idx] = w_r
        for idx in self.act_l: d.ctrl[idx] = w_l

    @staticmethod
    def _rate_limit(cur: float, tgt: float, max_rate: float, dt: float) -> float:
        return cur + np.clip(tgt - cur, -max_rate * dt, max_rate * dt)
