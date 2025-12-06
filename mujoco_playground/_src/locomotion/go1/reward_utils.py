from typing import Any

import jax
import jax.numpy as jp

__all__ = [
    "_reward_orientation",
    "_reward_pose",
    "_cost_torques",
    "_cost_action_rate",
    "_cost_joint_pos_limits",
    "_cost_dof_acc",
    "_cost_stay_still_rpy",
]



def _reward_orientation(
        vecA: jax.Array, 
        vecB: jax.Array
        ) -> jax.Array:
    """
    Cosine Distance between two unit vectors, normalized to [0, 1] and squared
    Input:
        - vecA: unit vector A
        - vecB: unit vector B
    Output:
        - reward
    """
    cos_dist = jp.dot(vecA, vecB)
    normalized = 0.5 * cos_dist + 0.5
    return jp.square(normalized)


def _reward_pose(
        pose1:      jax.Array, 
        pose2:      jax.Array, 
        isUpright:  jax.Array
        ) -> jax.Array:
    """
    L2 Loss between poses, exponentiated
    Loss computed only when Go1's orientation is upright
    Input:
        - pose1:        pose 1
        - pose2:        pose 2
        - isUpright:    Boolean gate on whether Go1 is upright
    Output:
        - reward
    """
    error = jp.sum(jp.square(pose1 - pose2))
    return jp.exp(-0.5 * error) * isUpright


def _cost_torques(
        torques: jax.Array
        ) -> jax.Array:
    """
    L2 Loss on torques
    Input:
        - torques:    torques applied
    Output:
        - reward
    """
    return jp.sum(jp.square(torques))


def _cost_action_rate(
        act:    jax.Array, 
        info:   dict[str, Any]
        ) -> jax.Array:
    """
    L2 Loss on action rate
    Input:
        - act:    current action
        - info:   state information dictionary
    Output:
        - reward
    """
    return jp.sum(jp.square(act - info["last_act"]))


def _cost_joint_pos_limits(
        soft_lowers:    jax.Array, 
        soft_uppers:    jax.Array, 
        qpos:           jax.Array
        ) -> jax.Array:
    """
    Cost on violations of joint position limits 
    Input:
        - soft_lowers:    lower limits
        - soft_uppers:    upper limits
        - qpos:           current joint positions
    Output:
        - reward
    """
    out_of_limits = -jp.clip(qpos - soft_lowers, None, 0.0)
    out_of_limits += jp.clip(qpos - soft_uppers, 0.0, None)
    return jp.sum(out_of_limits)


def _cost_dof_acc(
        qacc: jax.Array
        ) -> jax.Array:
    """
    L2 Loss on joint accelerations
    Input:
        - qacc:     joint accelerations
    Output:
        - reward
    """
    return jp.sum(jp.square(qacc))


def _cost_stay_still_rpy(
        qvel_rpy: jax.Array
        ) -> jax.Array:
    """
    L2 Loss on torso angular velocities (RPY)
    Input:
        - qvel_rpy: torso RPY velocities
    Output:
        - reward
    """
    return jp.sum(jp.square(qvel_rpy))