"""Shared terminal-event semantics for the minimal AEBS experiment."""

from typing import Optional, Tuple


SUCCESS = "success"
UNSAFE = "unsafe"
STOPPED_SAFE_OUTSIDE_GOAL = "stopped_safe_outside_goal"
OUT_OF_DOMAIN = "out_of_domain"
TIMEOUT = "timeout"

TERMINAL_OUTCOMES: Tuple[str, ...] = (
    SUCCESS,
    UNSAFE,
    STOPPED_SAFE_OUTSIDE_GOAL,
    OUT_OF_DOMAIN,
)
EPISODE_OUTCOMES: Tuple[str, ...] = TERMINAL_OUTCOMES + (TIMEOUT,)


def classify_terminal_outcome(
    distance_m: float,
    speed_mps: float,
    goal_distance_low_m: float = 5.0,
    goal_distance_high_m: float = 6.0,
    safe_speed_mps: float = 0.5,
    domain_distance_low_m: float = 5.0,
    domain_distance_high_m: float = 16.0,
    stopped_speed_mps: float = 1e-6,
) -> Optional[str]:
    """Return one mutually exclusive terminal outcome, or ``None``.

    Crossing the goal distance while still too fast is unsafe.  Reaching the
    goal interval at or below the safe speed is success.  A low-speed state
    below the modeled distance domain is kept distinct from success.
    """

    distance_m = float(distance_m)
    speed_mps = float(speed_mps)

    if distance_m <= goal_distance_high_m and speed_mps > safe_speed_mps:
        return UNSAFE
    if (
        goal_distance_low_m <= distance_m <= goal_distance_high_m
        and speed_mps <= safe_speed_mps
    ):
        return SUCCESS
    if distance_m < domain_distance_low_m or distance_m > domain_distance_high_m:
        return OUT_OF_DOMAIN
    if speed_mps <= stopped_speed_mps and distance_m > goal_distance_high_m:
        return STOPPED_SAFE_OUTSIDE_GOAL
    return None
