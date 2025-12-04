# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Zero Gravity Reorientation task for Go1."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.go1 import base as go1_base
from mujoco_playground._src.locomotion.go1 import go1_constants as consts


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.004,
      episode_length=500,
      Kp=35.0,
      Kd=0.5,
      action_repeat=1,
      action_scale=0.3,
      soft_joint_pos_limit_factor=0.9,
      init_from_crouch=0.0,
      energy_termination_threshold=np.inf,
      timeout_steps=3000,  # 60 seconds: 60 / 0.02 = 3000 steps  
      noise_config=config_dict.create(
          level=0.0,  # Set to 0.0 to disable noise. (Ideal condition)
          scales=config_dict.create(
              joint_pos=0.01,
              joint_vel=1.5,
              gyro=0.2,
              gravity=0.05,
              linvel=0.1,
          ),
      ),
      # Reward Configuration Scales
      reward_config=config_dict.create(
          scales=config_dict.create(
              orientation=2.0,         
              action_rate=0.0,        
              termination=-1.0,         # negative reward for unsuccessful termination
              dof_pos_limits=-1.0,      
              torques=0.0,            
              pose=0.0,
              stay_still=-0.01,          # penalize linear and angular velocity
              energy=0.0,
              dof_acc=0.0,            
          ),
      ),
      impl="jax",
      nconmax=30 * 8192,
      njmax=200,
  )


class Handstand(go1_base.Go1Env):
  """Landing task for Go1."""

  def __init__(
      self,
      config:           config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
      ) -> None:
    
    super().__init__(
        xml_path=consts.FULL_FLAT_TERRAIN_XML.as_posix(),
        config=config,
        config_overrides=config_overrides,
    )
    
    # NOTE: Zero Gravity
    self._mj_model.opt.gravity[:] = [0.0, 0.0, 0.0]
    self._mjx_model = mjx.put_model(self._mj_model, impl=self._config.impl)
    
    self._post_init()



  def _post_init(
      self
      ) -> None:
    
    self._init_q = jp.array(self._mj_model.keyframe("home").qpos)
    self._crouch_q = jp.array(self._mj_model.keyframe("pre_recovery").qpos)
    self._default_pose = jp.array(self._mj_model.keyframe("home").qpos[7:])

    self._lowers, self._uppers = self.mj_model.jnt_range[1:].T
    c = (self._lowers + self._uppers) / 2
    r = self._uppers - self._lowers
    self._soft_lowers = c - 0.5 * r * self._config.soft_joint_pos_limit_factor
    self._soft_uppers = c + 0.5 * r * self._config.soft_joint_pos_limit_factor

    self._torso_body_id = self._mj_model.body(consts.ROOT_BODY).id
    self._feet_site_id = np.array(
        [self._mj_model.site(name).id for name in consts.FEET_SITES]
    )
    self._floor_geom_id = self._mj_model.geom("floor").id
    self._feet_geom_id = np.array(
        [self._mj_model.geom(name).id for name in consts.FEET_GEOMS]
    )
    self._z_des = 0.55
    self._desired_up_vec = jp.array([0, 0, 1])

    self._joint_ids = jp.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    self._joint_pose = self._default_pose[self._joint_ids]

    geom_names = [
        "fl_calf1",
        "fl_calf2",
        "fr_calf1",
        "fr_calf2",
        "fl_thigh1",
        "fl_thigh2",
        "fl_thigh3",
        "fr_thigh1",
        "fr_thigh2",
        "fr_thigh3",
        "fl_hip",
        "fr_hip",
        "rl_calf1",
        "rl_calf2",
        "rr_calf1",
        "rr_calf2",
        "rl_thigh1",
        "rl_thigh2",
        "rl_thigh3",
        "rr_thigh1",
        "rr_thigh2",
        "rr_thigh3",
        "rl_hip",
        "rr_hip",
    ]

    self._unwanted_contact_geom_ids = np.array(
        [self._mj_model.geom(name).id for name in geom_names]
    )

    # Contact sensor ids.
    self._fullcollision_floor_found_sensor = [
        self._mj_model.sensor(f"{geom}_floor_found").id
        for geom in geom_names
    ]



  def _domain_randomize(
      self,
      rng: jax.Array
      ) -> None:
    
    # Quadruped Spawn Pose
    # - init from CROUCH or STANDING
    rng, key = jax.random.split(rng)
    if_init_from_crouch = jax.random.bernoulli(key, self._config.init_from_crouch)
    qpos = jp.where(if_init_from_crouch, self._crouch_q, self._init_q)

    # Quadruped Spawn Position
    # - x   +=U(-0.5, 0.5)
    # - y   +=U(-0.5, 0.5)
    # - z   +=U( 1.0, 1.5)
    rng, key = jax.random.split(rng)
    dxy = jax.random.uniform(key, (2,), minval=-0.5, maxval=0.5)
    qpos = qpos.at[0:2].set(qpos[0:2] + dxy)
    rng, key = jax.random.split(rng)
    dz = jax.random.uniform(key, minval=1.0, maxval=1.5)
    qpos = qpos.at[2].set(qpos[2] + dz)

    # Quadruped Spawn Orientation
    # - roll    = U(-3.14, 3.14)
    # - pitch   = U(-3.14, 3.14)
    # - yaw     = U(-3.14, 3.14)
    rng, key = jax.random.split(rng)
    rpy = jax.random.uniform(key, (3,), minval=-3.14, maxval=3.14)
    quat_roll =     math.axis_angle_to_quat(jp.array([1, 0, 0]), rpy[0])
    quat_pitch =    math.axis_angle_to_quat(jp.array([0, 1, 0]), rpy[1])
    quat_yaw =      math.axis_angle_to_quat(jp.array([0, 0, 1]), rpy[2])
    quat =      math.quat_mul(quat_yaw, math.quat_mul(quat_pitch, quat_roll))
    new_quat =  math.quat_mul(qpos[3:7], quat)
    qpos = qpos.at[3:7].set(new_quat)

    # Quadruped Spawn Velocity
    # dx        = U(-0.5, 0.5)
    # dy        = U(-0.5, 0.5)
    # dz        = U(-0.5, 0.5)
    # droll     = U(-0.5, 0.5)
    # dpitch    = U(-0.5, 0.5)
    # dyaw      = U(-0.5, 0.5)
    qvel_nonzero = jp.zeros(self.mjx_model.nv)
    rng, key = jax.random.split(rng)
    qvel_nonzero = qvel_nonzero.at[0:6].set(jax.random.uniform(key, (6,), minval=-0.5, maxval=0.5))
    qvel = jp.where(if_init_from_crouch, jp.zeros(self.mjx_model.nv), qvel_nonzero)

    return rng, qpos, qvel



  def reset(
      self,
      rng: jax.Array
      ) -> mjx_env.State:

    # Domain Randomization
    rng, qpos, qvel = self._domain_randomize(rng)

    # State - Data
    data = mjx_env.make_data(
        self.mj_model,
        qpos=qpos,
        qvel=qvel,
        ctrl=qpos[7:],
        impl=self.mjx_model.impl.value,
        nconmax=self._config.nconmax,
        njmax=self._config.njmax,
    )
    data = mjx.forward(self.mjx_model, data)

    # State - Info
    info = {
        "step": 0,
        "rng": rng,
        "last_act": jp.zeros(self.mjx_model.nu),
    }

    # State - Metrics
    metrics = {}
    for k in self._config.reward_config.scales.keys():
        metrics[f"reward/{k}"] = jp.zeros(())

    # State - Observation
    contact = jp.array([
        data.sensordata[self._mj_model.sensor_adr[sensorid]] > 0
        for sensorid in self._fullcollision_floor_found_sensor
    ])
    obs = self._get_obs(data, info, contact)

    # State - Reward & Termination
    reward, done = jp.zeros(2)

    return mjx_env.State(data, obs, reward, done, metrics, info)
  


  def step(
      self,
      state:    mjx_env.State,
      action:   jax.Array
      ) -> mjx_env.State:
    
    motor_targets = state.data.ctrl + action * self._config.action_scale

    # State - Data
    data = mjx_env.step(self.mjx_model, state.data, motor_targets, self.n_substeps)

    contact = jp.array([
        data.sensordata[self._mj_model.sensor_adr[sensorid]] > 0
        for sensorid in self._fullcollision_floor_found_sensor
    ])

    # State - Observation
    obs = self._get_obs(data, state.info, contact)

    # State - Termination
    terminate = self._get_termination(data, state.info, contact)
    terminate_timeout = self._get_termination_timeout(state.info)

    # State - Reward
    rewards = self._get_reward(data, action, state.info, terminate, terminate_timeout)
    rewards = {
        k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
    }
    reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)

    # State - Info & Metrics
    state.info["step"] += 1
    state.info["last_act"] = action
    for k, v in rewards.items():
      state.metrics[f"reward/{k}"] = v

    # Terminate on both bad and good terminations
    done = (terminate | terminate_timeout).astype(reward.dtype)
    state = state.replace(data=data, obs=obs, reward=reward, done=done)

    return state



  # Terminations
  def _get_termination(
      self,
      data:     mjx.Data,
      info:     dict[str, Any],
      contact:  jax.Array
      ) -> jax.Array:
    del info  # Unused.
    # fall_termination = self.get_upvector(data)[-1] < -0.25
    contact_termination = jp.any(contact)
    energy = jp.sum(jp.abs(data.actuator_force) * jp.abs(data.qvel[6:]))
    energy_termination = energy > self._config.energy_termination_threshold
    return contact_termination | energy_termination
  
  def _get_termination_timeout(
      self,
      info:     dict[str, Any],
    ) -> jax.Array:
    return info["step"] >= self._config.timeout_steps

  def _get_obs(
      self,
      data:     mjx.Data,
      info:     dict[str, Any],
      contact:  jax.Array
  ) -> Dict[str, jax.Array]:
    
    del contact  # Unused.

    gyro = self.get_gyro(data)
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_gyro = (
        gyro
        + (2 * jax.random.uniform(noise_rng, shape=gyro.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.gyro
    )

    gravity = self.get_gravity(data)
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_gravity = (
        gravity
        + (2 * jax.random.uniform(noise_rng, shape=gravity.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.gravity
    )

    joint_angles = data.qpos[7:]
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_joint_angles = (
        joint_angles
        + (2 * jax.random.uniform(noise_rng, shape=joint_angles.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.joint_pos
    )

    joint_vel = data.qvel[6:]
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_joint_vel = (
        joint_vel
        + (2 * jax.random.uniform(noise_rng, shape=joint_vel.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.joint_vel
    )

    linvel = self.get_local_linvel(data)
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_linvel = (
        linvel
        + (2 * jax.random.uniform(noise_rng, shape=linvel.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.linvel
    )

    state = jp.hstack([
        noisy_linvel,
        noisy_gyro,
        noisy_gravity,
        noisy_joint_angles - self._default_pose,
        noisy_joint_vel,
        info["last_act"],
    ])

    accelerometer = self.get_accelerometer(data)
    linvel = self.get_local_linvel(data)
    angvel = self.get_global_angvel(data)
    torso_height = data.site_xpos[self._imu_site_id][2]

    privileged_state = jp.hstack([
        state,
        gyro,
        accelerometer,
        linvel,
        angvel,
        joint_angles,
        joint_vel,
        data.actuator_force,
        torso_height,
    ])

    return {
        "state": state,
        "privileged_state": privileged_state,
    }



  def _get_reward(
      self,
      data:     mjx.Data,
      action:   jax.Array,
      info:     dict[str, Any],
      terminate:     jax.Array,
      terminate_timeout: jax.Array,
      ) -> dict[str, jax.Array]:
    
    up_vector = data.site_xmat[self._imu_site_id] @ jp.array([0.0, 0.0, 1.0])
    # up_vector = self.get_upvector(data)
    
    joint_torques = data.actuator_force
    
    done_bad = (terminate & (~terminate_timeout)).astype(jp.float32)
    done_timeout = (terminate_timeout).astype(jp.float32)
    

    rewards = {
        "orientation":          self._reward_orientation(up_vector, self._desired_up_vec),
        "action_rate":          self._cost_action_rate(action, info),
        "torques":              self._cost_torques(joint_torques),
        "termination":          done_bad,  # collision/energy fail (negative reward)
        "dof_pos_limits":       self._cost_joint_pos_limits(data.qpos[7:]),
        "dof_acc":              self._cost_dof_acc(data.qacc[6:]),
        "pose":                 self._cost_pose(data.qpos[7:]),
        "stay_still":           self._cost_stay_still(data.qvel[:6]),
        "energy":               self._cost_energy(data.qvel[6:], data.actuator_force),
    }

    return rewards



  # Task-Specific Rewards
  def _cost_stay_still(self, qvel: jax.Array) -> jax.Array:
    return jp.sum(jp.square(qvel[:3])) + jp.sum(jp.square(qvel[3:6]))

  def _reward_orientation(
      self, vecA: jax.Array, vecB: jax.Array
  ) -> jax.Array:
    cos_dist = jp.dot(vecA, vecB)
    normalized = 0.5 * cos_dist + 0.5
    return jp.square(normalized)

  def _cost_pose(self, qpos: jax.Array) -> jax.Array:
    return jp.sum(jp.square(qpos[self._joint_ids] - self._joint_pose))



  # General Rewards
  def _cost_torques(self, torques: jax.Array) -> jax.Array:
    return jp.sum(jp.square(torques))

  def _cost_energy(
      self, qvel: jax.Array, qfrc_actuator: jax.Array
  ) -> jax.Array:
    return jp.sum(jp.abs(qvel) * jp.abs(qfrc_actuator))

  def _cost_action_rate(
      self, act: jax.Array, info: dict[str, Any]
  ) -> jax.Array:
    return jp.sum(jp.square(act - info["last_act"]))

  def _cost_joint_pos_limits(self, qpos: jax.Array) -> jax.Array:
    out_of_limits = -jp.clip(qpos - self._soft_lowers, None, 0.0)
    out_of_limits += jp.clip(qpos - self._soft_uppers, 0.0, None)
    return jp.sum(out_of_limits)

  def _cost_dof_acc(self, qacc: jax.Array) -> jax.Array:
    return jp.sum(jp.square(qacc))



class Footstand(Handstand):
  """Footstand task for Go1."""

  def _post_init(self) -> None:
    super()._post_init()

    self._handstand_pose = jp.array(
        self._mj_model.keyframe("footstand").qpos[7:]
    )
    self._handstand_q = jp.array(self._mj_model.keyframe("footstand").qpos)
    self._joint_ids = jp.array([0, 1, 2, 3, 4, 5])
    self._joint_pose = self._default_pose[self._joint_ids]
    self._desired_forward_vec = jp.array([0, 0, 1])
    self._z_des = 0.53

    geom_names = [
        "rl_calf1",
        "rl_calf2",
        "rr_calf1",
        "rr_calf2",
        "rl_thigh1",
        "rl_thigh2",
        "rl_thigh3",
        "rr_thigh1",
        "rr_thigh2",
        "rr_thigh3",
        "rl_hip",
        "rr_hip",
    ]
    self._unwanted_contact_geom_ids = np.array(
        [self._mj_model.geom(name).id for name in geom_names]
    )

    feet_geom_names = ["FR", "FL"]
    self._feet_geom_ids = np.array(
        [self._mj_model.geom(name).id for name in feet_geom_names]
    )