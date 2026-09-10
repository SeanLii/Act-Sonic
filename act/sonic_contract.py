"""The Unitree G1 SONIC observation/action contract.

The ordering mirrors ``unitree_g1_sonic`` in Isaac-GR00T.  LeRobot stores
state and action groups concatenated in this order, while the live SONIC
client sends and receives a dictionary with one array per group.
"""

from __future__ import annotations


SONIC_VIDEO_KEY = "ego_view"
SONIC_LANGUAGE_KEY = "annotation.human.task_description"
SONIC_ACTION_HORIZON = 40

# (field name, width) in the order used by observation.state.
SONIC_STATE_FIELDS: tuple[tuple[str, int], ...] = (
    ("left_leg", 6),
    ("right_leg", 6),
    ("waist", 3),
    ("left_arm", 7),
    ("right_arm", 7),
    ("left_hand", 7),
    ("right_hand", 7),
    ("projected_gravity", 3),
)

# SONIC latent protocol v4: a 64-D whole-body token and two 7-D hands.
SONIC_ACTION_FIELDS: tuple[tuple[str, int], ...] = (
    ("motion_token", 64),
    ("left_hand_joints", 7),
    ("right_hand_joints", 7),
)

SONIC_QPOS_DIM = sum(width for _, width in SONIC_STATE_FIELDS)
SONIC_ACTION_DIM = sum(width for _, width in SONIC_ACTION_FIELDS)


def contract_dict(action_horizon: int = SONIC_ACTION_HORIZON) -> dict[str, object]:
    """Return a JSON-serializable checkpoint description of the wire contract."""
    return {
        "version": 1,
        "video_key": SONIC_VIDEO_KEY,
        "language_key": SONIC_LANGUAGE_KEY,
        "state_fields": [list(field) for field in SONIC_STATE_FIELDS],
        "action_fields": [list(field) for field in SONIC_ACTION_FIELDS],
        "action_horizon": action_horizon,
    }
