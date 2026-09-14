from abc import ABC, abstractmethod


class BaseController(ABC):
    """Classical local controller commanding body velocity (v, omega).
    Exposes an SB3-compatible predict() so it is interchangeable with SAC."""

    deterministic = True

    @abstractmethod
    def predict(self, obs_dict):
        raise NotImplementedError

    def reset(self):
        """Reset per-episode state (warm-starts, filters). No-op by default."""
        return None
