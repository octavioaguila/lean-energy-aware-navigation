import casadi as ca
import numpy as np
import os
import sys
import json
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from baselines.base_controller import BaseController

# Smooth approximation of abs(x) for IPOPT (C2-smooth, avoids undefined 2nd
# derivative at x=0 that causes Error_In_Step_Computation with ca.fabs).
_EPS = 1e-6
def smooth_abs(x):
    return ca.sqrt(x**2 + _EPS)

# Smoothing width for Coulomb friction — matches bunker_velocity_controller.py
EPSILON_W = 0.05  # rad/s


class BunkerNMPC(BaseController):
    """
    Non-Linear Model Predictive Controller for the Bunker robot.

    Uses a 5-state dynamic model  x = [x_w, y_w, theta, v, omega]
    with torque inputs  u = [tau_R, tau_L]  identified from real grass-terrain
    data (motors_params.json / controller_calibration.json).

    Energy is computed from the same physics-grounded model used in
    BunkerVelocityController, making the optimisation consistent with the
    actual power estimator.
    """

    def __init__(self, env, look_ahead_horizon=20, n_nearest_lidar=20):
        self.env = env
        self.n_nearest_lidar = n_nearest_lidar
        self.look_ahead_horizon = look_ahead_horizon
        self.dt = env.model.opt.timestep * env.frame_skip

        self.v_max = env.v_max
        self.w_max = env.w_max
        self.energy_weight = env.energy_weight

        # Load identified physics constants from JSON files
        _root       = os.path.join(os.path.dirname(__file__), '..')
        _motors_f   = os.path.join(_root, 'config', 'motors_params.json')
        _calib_f    = os.path.join(_root, 'config', 'controller_calibration.json')

        with open(_motors_f) as f:
            mp = json.load(f)
        self.kt_r       = mp['K_R_sprocket']   # Nm/A at sprocket shaft
        self.kt_l       = mp['K_L_sprocket']   # Nm/A at sprocket shaft
        self.tau_c      = mp['tau_c']          # Nm  Coulomb scrubbing torque
        self.w_eff      = mp['W_eff']          # m   effective track width
        self.c_damp     = mp['c']              # N·s/m  viscous translation damping
        self.b_damp     = mp['b']              # N·m·s/rad  viscous yaw damping
        self.r_sprocket = mp['r_sprocket']     # m
        self.mass       = mp['m']              # kg
        self.J_yaw      = mp['J']             # kg·m²

        with open(_calib_f) as f:
            cb = json.load(f)
        self.r_terminal      = cb['R_terminal']
        self.fidelity_factor = cb['fidelity_factor']

        # Max torque bound (Nm).  Derived from worst-case navigation load:
        # at v_max + w_max the dominant yaw demand is ≈6 Nm per side; 20 Nm
        # gives headroom for accelerations while keeping the NLP well-posed.
        self.tau_max = 20.0

        # Build NLP (JIT-compiled with CasADi / IPOPT)
        self._setup_nlp()

        # Warm-start bookkeeping
        self._z_init     = None
        self._solve_count = 0
        self._fail_count  = 0

    def _setup_nlp(self):
        """
        Build the NMPC optimisation problem.

        Decision variables: X (5×(N+1)), U (2×N), S (N)
        Parameters:         start state (5), goal (2), obstacle points (2×M)
        """
        N  = self.look_ahead_horizon
        M  = self.n_nearest_lidar
        dt = self.dt

        nx_state = 5            # [x_w, y_w, theta, v, omega]
        nx = nx_state * (N + 1)
        nu = 2 * N              # [tau_R, tau_L] per step
        ns = N                  # one slack per horizon step

        self._N  = N
        self._M  = M
        self._nx = nx
        self._nu = nu
        self._ns = ns

        # Bake physics constants as Python floats so the JIT compiler can fold them
        m      = float(self.mass)
        r      = float(self.r_sprocket)
        W      = float(self.w_eff)
        J      = float(self.J_yaw)
        c      = float(self.c_damp)
        b      = float(self.b_damp)
        tau_c  = float(self.tau_c)
        kt_r   = float(self.kt_r)
        kt_l   = float(self.kt_l)
        R_term = float(self.r_terminal)
        ff     = float(self.fidelity_factor)
        eps_w  = float(EPSILON_W)

        # ── Decision variables ────────────────────────────────────────────────
        X = ca.MX.sym('X', nx)   # 5×(N+1) robot state
        U = ca.MX.sym('U', nu)   # 2×N torque inputs
        S = ca.MX.sym('S', ns)   # N obstacle slack variables
        z = ca.vertcat(X, U, S)

        # ── Parameters (fixed at each solve call) ─────────────────────────────
        P_start = ca.MX.sym('P_start', 5)    # [x, y, theta, v, omega]
        P_goal  = ca.MX.sym('P_goal',  2)    # [x_g, y_g]
        P_obs   = ca.MX.sym('P_obs', 2, M)   # obstacle points

        # Total parameter vector (same length as old formulation: 5+2+2M = 7+2M)
        P = ca.vertcat(P_start, P_goal, ca.reshape(P_obs, -1, 1))

        # ── Reshape into matrices (column-major) ──────────────────────────────
        X_mat = ca.reshape(X, nx_state, N + 1)  # (5, N+1)
        U_mat = ca.reshape(U, 2, N)             # (2, N)

        obj = 0
        g   = []

        # Initial state equality: X[:,0] == P_start
        g.append(X_mat[:, 0] - P_start)

        # ── Stage loop ────────────────────────────────────────────────────────
        for k in range(N):
            st  = X_mat[:, k]
            con = U_mat[:, k]

            xk, yk, thetak = st[0], st[1], st[2]
            vk, omk        = st[3], st[4]
            tau_r, tau_l   = con[0], con[1]

            # ── 5-state forward Euler dynamics ────────────────────────────────
            # Translational:  m·v̇  = (τ_R + τ_L)/r  − c·v
            # Rotational:     J·ω̇  = (τ_R − τ_L)·W/(2r) − b·ω − τ_c·tanh(ω/ε)
            dv_dt  = (tau_r + tau_l) / (m * r) - (c / m) * vk
            dom_dt = (tau_r - tau_l) * W / (2.0 * J * r) \
                     - (b / J) * omk \
                     - (tau_c / J) * ca.tanh(omk / eps_w)

            next_st = ca.vertcat(
                xk     + vk * ca.cos(thetak) * dt,
                yk     + vk * ca.sin(thetak) * dt,
                thetak + omk * dt,
                vk     + dv_dt  * dt,
                omk    + dom_dt * dt,
            )
            g.append(X_mat[:, k + 1] - next_st)

            # Goal-distance stage cost
            dist_cost = ca.sumsqr(X_mat[:2, k] - P_goal)

            # Physics-grounded energy estimate
            # Sprocket angular velocities from predicted body state
            Omega_R = (vk + omk * W / 2.0) / r
            Omega_L = (vk - omk * W / 2.0) / r

            # Motor currents implied by the applied torques (τ = K · I)
            I_R = tau_r / kt_r
            I_L = tau_l / kt_l

            # Mechanical power:  P_mech = |τ_R·Ω_R| + |τ_L·Ω_L|
            p_mech = smooth_abs(tau_r * Omega_R) + smooth_abs(tau_l * Omega_L)

            # Electrical (ohmic) heat:  P_heat = R · (I_R² + I_L²)
            p_heat = R_term * (I_R**2 + I_L**2)

            # Coulomb scrubbing:  P_scrub = τ_c · |tanh(ω/ε)|
            p_coulomb = tau_c * smooth_abs(ca.tanh(omk / eps_w))

            energy_watts = (p_mech + p_heat + p_coulomb) * ff

            obj += 500.0 * dist_cost + self.energy_weight * energy_watts * dt

        # Terminal cost
        obj += 1_000.0 * ca.sumsqr(X_mat[:2, N] - P_goal)

        # Obstacle slack penalties
        obstacle_slack_linear    = 1_000.0
        obstacle_slack_quadratic = 10_000.0
        obj += obstacle_slack_linear * ca.sum1(S) \
             + obstacle_slack_quadratic * ca.sumsqr(S)

        # Obstacle constraints (vectorised)
        X_pos = X_mat[:2, 1:]  # (2, N) positions along horizon (skip k=0)

        xk_h = ca.transpose(X_pos[0, :])  # (N,)
        yk_h = ca.transpose(X_pos[1, :])  # (N,)

        obs_x = P_obs[0, :]
        obs_y = P_obs[1, :]

        dx      = ca.repmat(xk_h, 1, M) - ca.repmat(obs_x, N, 1)
        dy      = ca.repmat(yk_h, 1, M) - ca.repmat(obs_y, N, 1)
        dist_sq = dx**2 + dy**2         # (N, M)

        S_rep  = ca.repmat(S, 1, M)     # (N, M)
        d_safe = 0.75                   # minimum clearance (m)
        g_obs  = ca.reshape(dist_sq + S_rep, N * M, 1) - d_safe**2
        g.append(g_obs)

        # Slack non-negativity: S[k] >= 0
        g.append(S)

        g_vec = ca.vertcat(*g)

        nlp = {'x': z, 'p': P, 'f': obj, 'g': g_vec}

        opts = {
            'jit':                           True,
            'compiler':                      'shell',
            'print_time':                    0,
            'ipopt.print_level':             0,
            'ipopt.tol':                     1e-3,
            'ipopt.acceptable_tol':          1e-2,
            'ipopt.max_iter':                40,
            'ipopt.mu_strategy':             'adaptive',
            'ipopt.linear_solver':           'mumps',
            'ipopt.warm_start_init_point':   'yes',
        }

        self.solver = ca.nlpsol('bunker_nmpc_solver', 'ipopt', nlp, opts)

        # Pre-compute decision-variable bounds
        nz = nx + nu + ns
        self._lbx = -np.inf * np.ones(nz)
        self._ubx =  np.inf * np.ones(nz)

        # Torque bounds on U block
        for k in range(N):
            idx_r = nx + 2 * k
            idx_l = nx + 2 * k + 1
            self._lbx[idx_r] = -self.tau_max
            self._ubx[idx_r] =  self.tau_max
            self._lbx[idx_l] = -self.tau_max
            self._ubx[idx_l] =  self.tau_max

        # Velocity bounds on state X[3,:] (v) and X[4,:] (omega) for k=1..N.
        # k=0 is fixed by the initial-state equality constraint, so not bounded here.
        # Column-major layout: X_mat[row, col] → flat index = nx_state*col + row
        for k in range(1, N + 1):
            idx_v = nx_state * k + 3
            idx_w = nx_state * k + 4
            self._lbx[idx_v] = -self.v_max
            self._ubx[idx_v] =  self.v_max
            self._lbx[idx_w] = -self.w_max
            self._ubx[idx_w] =  self.w_max

        # Pre-compute constraint bounds
        g_eq_len  = nx_state * (N + 1)   # initial + N dynamics equalities
        g_obs_len = N * M
        g_s_len   = N

        self._g_eq_len  = g_eq_len
        self._g_obs_len = g_obs_len
        self._g_s_len   = g_s_len

        lbg = np.zeros(g_eq_len + g_obs_len + g_s_len)
        ubg = np.zeros_like(lbg)

        ubg[g_eq_len : g_eq_len + g_obs_len] = np.inf   # obstacle: >= 0
        ubg[g_eq_len + g_obs_len :]           = np.inf   # slack:    >= 0

        self._lbg = lbg
        self._ubg = ubg

    def _build_initial_guess(self, curr_pos, goal_pos):
        """Straight-line state guess with steady-state torque inputs."""
        N  = self._N
        nx = self._nx
        nu = self._nu
        ns = self._ns

        x0, y0, theta0 = curr_pos
        gx, gy = goal_pos

        dist = float(np.hypot(gx - x0, gy - y0))
        theta_goal = float(np.arctan2(gy - y0, gx - x0))

        v_guess = min(dist / max(N * self.dt, 1e-6), self.v_max * 0.5)
        heading_err = float(np.arctan2(
            np.sin(theta_goal - theta0), np.cos(theta_goal - theta0)))
        w_guess = float(np.clip(
            heading_err / max(N * self.dt, 1e-6),
            -self.w_max * 0.5, self.w_max * 0.5))

        # State guess: straight-line position, constant (v_guess, w_guess)
        X_guess = np.zeros((5, N + 1))
        for k in range(N + 1):
            frac = k / N
            X_guess[0, k] = x0 + frac * (gx - x0)
            X_guess[1, k] = y0 + frac * (gy - y0)
            X_guess[2, k] = theta_goal
            X_guess[3, k] = v_guess
            X_guess[4, k] = w_guess

        # Torque guess: steady-state solution for (v_guess, w_guess)
        # τ_R + τ_L = c · v · r       (translation balance)
        # τ_R − τ_L = (b·ω + τ_c·tanh(ω/ε)) · 2r/W  (rotation balance)
        r = self.r_sprocket
        W = self.w_eff
        tau_sum  = self.c_damp * v_guess * r
        tau_diff = (self.b_damp * w_guess
                    + self.tau_c * float(np.tanh(w_guess / EPSILON_W))
                   ) * 2.0 * r / W
        tau_r_ss = (tau_sum + tau_diff) / 2.0
        tau_l_ss = (tau_sum - tau_diff) / 2.0

        U_guess = np.zeros((2, N))
        U_guess[0, :] = np.clip(tau_r_ss, -self.tau_max, self.tau_max)
        U_guess[1, :] = np.clip(tau_l_ss, -self.tau_max, self.tau_max)

        z0 = np.zeros(nx + nu + ns)
        z0[:nx]      = X_guess.reshape(-1, order='F')
        z0[nx:nx+nu] = U_guess.reshape(-1, order='F')
        # slack starts at zero

        return z0

    def reset(self):
        self._z_init = None

    def predict(self, obs_dict):
        """Process environment observation and return NMPC command.

        Signature mirrors stable_baselines3 `.predict()`:
            action, _ = controller.predict(obs_dict)
        """
        curr_pos = obs_dict['achieved_goal'][0]
        goal_pos = obs_dict['desired_goal'][0][:2]
        raw_obs  = obs_dict['observation'][0]

        # ── LiDAR: select the M nearest obstacle points ───────────────────────
        lidar_flat = raw_obs[:self.env.n_lidar * 3]
        lidar_pts  = lidar_flat.reshape(self.env.n_lidar, 3)

        d_norms   = lidar_pts[:, 2]
        raw_dists = (d_norms + 1.0) / 2.0 * self.env.lidar_max_range

        closest_idx = np.argsort(raw_dists)[:self.n_nearest_lidar]

        obstacles = []
        for i in closest_idx:
            sin_a, cos_a, d_norm = lidar_pts[i]
            dist = raw_dists[i]
            if d_norm < -0.2:   # only points closer than ~40 % of max range
                lx = curr_pos[0] + dist * (
                    cos_a * np.cos(curr_pos[2]) - sin_a * np.sin(curr_pos[2]))
                ly = curr_pos[1] + dist * (
                    cos_a * np.sin(curr_pos[2]) + sin_a * np.cos(curr_pos[2]))
                obstacles.append([lx, ly])

        obs_array = np.ones((2, self.n_nearest_lidar)) * 99.0
        for i, ob in enumerate(obstacles[:self.n_nearest_lidar]):
            obs_array[:, i] = ob

        # ── Current body-frame velocities from observation ────────────────────
        v_norm = raw_obs[self.env.n_lidar * 3]
        w_norm = raw_obs[self.env.n_lidar * 3 + 1]
        v_cur  = float(v_norm * self.v_max)
        w_cur  = float(w_norm * self.w_max)

        # ── Parameter vector: [x, y, theta, v, omega, x_g, y_g, obs(flat)] ───
        p_start = np.array([
            float(curr_pos[0]), float(curr_pos[1]), float(curr_pos[2]),
            v_cur, w_cur,
        ], dtype=float)
        p_goal  = np.asarray(goal_pos, dtype=float).flatten()
        p_obs   = obs_array.astype(float).reshape(-1, order='F')
        p = np.concatenate([p_start, p_goal, p_obs])

        # ── Warm-start / initial guess ────────────────────────────────────────
        if self._z_init is None:
            z0 = self._build_initial_guess(curr_pos, goal_pos)
        else:
            z0 = self._z_init.copy()
            nx = self._nx
            nu = self._nu
            N  = self._N

            X_prev = z0[:nx].reshape((5, N + 1), order='F')
            U_prev = z0[nx:nx+nu].reshape((2, N),   order='F')
            S_prev = z0[nx+nu:].copy()

            # Update initial state with current measurement
            X_prev[:, 0] = p_start

            # Shift control sequence forward by one step
            U_init         = np.zeros_like(U_prev)
            U_init[:, :-1] = U_prev[:, 1:]
            U_init[:, -1]  = U_prev[:, -1]

            # Shift slack forward
            S_init      = np.zeros_like(S_prev)
            S_init[:-1] = S_prev[1:]
            S_init[-1]  = S_prev[-1]

            z0[:nx]      = X_prev.reshape(-1, order='F')
            z0[nx:nx+nu] = U_init.reshape(-1, order='F')
            z0[nx+nu:]   = S_init

        # Solve NLP
        try:
            t_start = time.time()
            sol = self.solver(
                x0=z0, p=p,
                lbx=self._lbx, ubx=self._ubx,
                lbg=self._lbg, ubg=self._ubg,
            )
            solve_time = time.time() - t_start

            z_opt = np.array(sol['x']).flatten()
            self._z_init     = z_opt
            self._fail_count  = 0
            self._solve_count += 1

            # Extract predicted next-step velocities from state trajectory
            # v₁* = X[3, 1],  ω₁* = X[4, 1]  (column-major reshape)
            nx    = self._nx
            N     = self._N
            X_opt = z_opt[:nx].reshape((5, N + 1), order='F')
            v0    = float(X_opt[3, 1])
            w0    = float(X_opt[4, 1])

            # Normalise to [-1, 1] for env compatibility
            action = np.array([v0 / self.v_max, w0 / self.w_max], dtype=float)

        except KeyboardInterrupt:
            raise
        except Exception as e:
            self._fail_count += 1
            print(f'NMPC solver failed (failure #{self._fail_count}): {e}')
            action = np.array([0.0, 0.0], dtype=float)

        if obs_dict['observation'].ndim > 1:
            return np.expand_dims(action, axis=0), None
        return action, None
