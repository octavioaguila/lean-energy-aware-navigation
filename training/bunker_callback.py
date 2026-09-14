import os
from datetime import datetime
from stable_baselines3.common.callbacks import BaseCallback
from collections import deque
import numpy as np

class BunkerCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.collision_buffer = deque(maxlen=100)
        self.energy_buffer = deque(maxlen=100)

    def _on_step(self) -> bool:
        for i, done in enumerate(self.locals["dones"]):
            if done:
                info = self.locals["infos"][i]
                if "collision" in info:
                    self.collision_buffer.append(float(info["collision"]))
        return True

    def _on_rollout_end(self) -> None:
        if len(self.collision_buffer) > 0:
            self.logger.record("rollout/collision_rate", np.mean(self.collision_buffer))


class SmoothedSaveCallback(BaseCallback):
    """
    Plugs into EvalCallback via callback_after_eval.
    Saves the model when the EMA-smoothed mean reward beats the best seen so far.
    Uses the same EMA factor as TensorBoard's default smoothing (alpha=0.6).
    """

    def __init__(self, model_save_path: str, ema_alpha: float = 0.6,
                 log_dir: str | None = None, verbose: int = 1):
        super().__init__(verbose)
        self.model_save_path = model_save_path
        self.ema_alpha = ema_alpha
        self._eval_log_path = os.path.join(log_dir, "eval_log.txt") if log_dir is not None else None
        self._smoothed: float | None = None
        self._best_smoothed = -np.inf

    def _on_step(self) -> bool:
        eval_callback = self.parent
        mean_reward = float(np.mean(eval_callback.evaluations_results[-1]))

        if self._smoothed is None:
            self._smoothed = mean_reward
        else:
            self._smoothed = self.ema_alpha * self._smoothed + (1 - self.ema_alpha) * mean_reward

        model_saved = False
        if self._smoothed > self._best_smoothed:
            self._best_smoothed = self._smoothed
            os.makedirs(self.model_save_path, exist_ok=True)
            self.model.save(os.path.join(self.model_save_path, "best_model"))
            model_saved = True
            if self.verbose:
                print(f"[SmoothedSave] New best EMA reward: {self._smoothed:.2f} "
                      f"(raw: {mean_reward:.2f}) → saved")

        if self._eval_log_path is not None:
            step = eval_callback.evaluations_timesteps[-1]
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            saved_tag = " ◆ MODEL SAVED" if model_saved else ""
            with open(self._eval_log_path, "a") as f:
                f.write(f"[{ts}] step={step:>8d} | "
                        f"mean_reward={mean_reward:>8.2f} | "
                        f"ema_reward={self._smoothed:>8.2f}{saved_tag}\n")
        return True
