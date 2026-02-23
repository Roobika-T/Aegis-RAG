import math
import random


def laplace_noise(scale: float) -> float:
    """
    Minimal Laplace noise generator for DP-style analytics.
    Not for production use.
    """
    u = random.random() - 0.5
    return -scale * math.copysign(1.0, u) * math.log(1 - 2 * abs(u))


def noisy_count(true_count: int, epsilon: float = 1.0, sensitivity: int = 1) -> float:
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")
    scale = sensitivity / epsilon
    return true_count + laplace_noise(scale)

