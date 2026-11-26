"""Speed-aware timing baseline using the same observation as learned policies."""

import numpy as np

from env.dino import DUCK_HEIGHT, OBSERVATION_SCALES, STANDING_HEIGHT, Action


def timing_policy(observation: np.ndarray) -> Action:
    """Duck beneath middle hazards and initiate a jump near low hazards."""
    # Decode the runner height, speed, and first obstacle from the fixed scales.
    encoded = observation[:7]
    values = OBSERVATION_SCALES[:7] * encoded / (1 - np.abs(encoded))
    height, _, speed, gap, bottom, width, _ = values

    if width == 0 or bottom >= STANDING_HEIGHT:
        return Action.RUN
    if bottom > DUCK_HEIGHT:
        if gap <= speed:
            return Action.DUCK
    elif height == 0 and 0 <= gap <= 2.7 * speed:
        return Action.JUMP
    return Action.RUN
