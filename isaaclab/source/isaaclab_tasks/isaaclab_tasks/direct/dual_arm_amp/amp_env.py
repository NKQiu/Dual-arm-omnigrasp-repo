# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import torch
import os
import numpy as np
import gym
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.torch.transformations import tf_combine, tf_inverse, tf_vector
from pxr import UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import sample_uniform
from .amp_env_cfg import AmpMotionImitatorEnvCfg

from collections.abc import Sequence
from isaaclab.controllers.joint_impedance import JointImpedanceController, JointImpedanceControllerCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.sensors import ContactSensorCfg,ContactSensor
import math




class AmpMotionImitatorEnv(DirectRLEnv):
    #execution order
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: AmpMotionImitatorEnvCfg

    def __init__(self, cfg: AmpMotionImitatorEnvCfg, render_mode: str | None = None, **kwargs):
    
        super().__init__(cfg, render_mode, **kwargs)

        self.arm_joint_names = ["panda_joint.*"]
        self.arm_joint_ids = self.left_robot.find_joints(self.arm_joint_names)[0]

        limits_left = self.left_robot.data.soft_joint_pos_limits[:, self.arm_joint_ids, :]
        limits_right = self.right_robot.data.soft_joint_pos_limits[:, self.arm_joint_ids, :]
        all_limits = torch.cat([limits_left, limits_right], dim=1)   # [num_envs, 14, 2]
        dof_lower_limits = all_limits[0, :, 0].to(self.device)
        dof_upper_limits = all_limits[0, :, 1].to(self.device)

        self.action_offset = 0.5 * (dof_upper_limits + dof_lower_limits)
        self.action_scale = 0.5 * (dof_upper_limits - dof_lower_limits)

        # joint_pos and vel
        joint_pos_left = self.left_robot.data.joint_pos   # shape: [num_envs, num_joints]
        joint_vel_left = self.left_robot.data.joint_vel
        joint_pos_right = self.right_robot.data.joint_pos
        joint_vel_right = self.right_robot.data.joint_vel
        self.joint_pos = torch.cat([joint_pos_left, joint_pos_right], dim=1)   
        self.joint_vel = torch.cat([joint_vel_left, joint_vel_right], dim=1)
        
        jp_cfg = JointImpedanceControllerCfg(
            command_type="p_abs",   # p_rel
            impedance_mode="fixed",
            stiffness=400.0,
            damping_ratio=1.0,
            gravity_compensation=True,
            inertial_compensation=True,

        )
        self.left_controller=JointImpedanceController(jp_cfg, num_robots=self.scene.num_envs, dof_pos_limits=self.left_robot.data.soft_joint_pos_limits, device=self.sim.device)
        self.right_controller=JointImpedanceController(jp_cfg, num_robots=self.scene.num_envs, dof_pos_limits=self.right_robot.data.soft_joint_pos_limits, device=self.sim.device)


        # load expert data
        self.expert_trajectories: list[torch.Tensor] = []
        self.motion_lengths: list[int] = []
        if not hasattr(self.cfg, "motion_files") or not isinstance(self.cfg.motion_files, list) or not self.cfg.motion_files:
            raise ValueError("Configuration 'motion_files' must be defined as a non-empty list of .pt file paths.")

        print(f"Loading expert trajectories from: {self.cfg.motion_files}") 
        
        for pt_file_path_str in self.cfg.motion_files:
            # Handle potential relative paths from config
            pt_file_path = os.path.expanduser(pt_file_path_str) # Expand ~ user symbol
            if not os.path.isabs(pt_file_path):
                 # Attempt to make path relative to the environment file's directory if not absolute
                 # This might need adjustment based on your project structure / config file location
                 env_dir = os.path.dirname(__file__)
                 potential_path = os.path.join(env_dir, pt_file_path)
                 if os.path.exists(potential_path):
                      pt_file_path = potential_path
                 # else: assume path is correct relative to execution dir or is absolute

            print(f"Attempting to load: {pt_file_path}")
            try:
                trajectory_tensor = torch.load(pt_file_path, map_location=self.device)

                if not isinstance(trajectory_tensor, torch.Tensor):
                    print(f"Warning: File {pt_file_path} did not contain a valid PyTorch Tensor, skipping.")
                    continue
                if trajectory_tensor.ndim != 2 or trajectory_tensor.shape[1] != 28:
                    print(f"Warning: Tensor in {pt_file_path} has incorrect shape {trajectory_tensor.shape} (expected [T, 28]), skipping.")
                    continue
                if len(trajectory_tensor) == 0:
                    print(f"Warning: Tensor in {pt_file_path} has length 0, skipping.")
                    continue

                trajectory_tensor = trajectory_tensor.to(dtype=torch.float32)
                self.expert_trajectories.append(trajectory_tensor)
                self.motion_lengths.append(len(trajectory_tensor))
                print(f"Successfully loaded trajectory: {pt_file_path}, Length: {len(trajectory_tensor)}")

            except FileNotFoundError:
                print(f"Error: Expert motion file not found at: {pt_file_path}, skipping.")
            except Exception as e:
                print(f"Error loading or processing file {pt_file_path}: {e}, skipping.")

        if not self.expert_trajectories:
            raise ValueError("Failed to load any valid expert motion trajectories. Check 'motion_files' config and file contents.")

        self.num_trajectories = len(self.expert_trajectories)
        self.motion_lengths_tensor = torch.tensor(self.motion_lengths, dtype=torch.long, device=self.device) # Use long for indexing

        # --- Reference State Tracking (Multi-Trajectory) ---
        self.current_trajectory_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.current_frame_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # Buffers for next reference state (for policy obs and task reward)
        self.ref_q_next = torch.zeros((self.num_envs, 14), device=self.device)
        self.ref_qd_next = torch.zeros((self.num_envs, 14), device=self.device)       
                       
     

        # self.step_numbers= torch.zeros(self.scene.num_envs,device=self.sim.device).unsqueeze(1)
        
        # MotionLoader
        # --- AMP ----

        # num_amp_observations，表示历史长度

        self.single_amp_observation_space = 28   # single step amp observation
        if not hasattr(self.cfg, "num_amp_observations") or self.cfg.num_amp_observations <= 0:
            raise ValueError("configuration num_amp_observations must be defined for AMP")
        # amp_obs total dimenstions    
        self.amp_observation_size = self.cfg.num_amp_observations * self.single_amp_observation_space

        # define AMP observation space
        self.amp_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.amp_observation_size,))
        # amp buffer to store history
        self.amp_observation_buffer = torch.zeros(
            (self.num_envs, self.cfg.num_amp_observations, self.single_amp_observation_space), device=self.device
        )       
        self.extras = {}
        

    def _setup_scene(self):
        self.left_robot = Articulation(self.cfg.left_robot)
        self.right_robot = Articulation(self.cfg.right_robot)
        self.pillar1 = RigidObject(self.cfg.pillar1)
        self.pillar2= RigidObject(self.cfg.pillar2)
        self.table = RigidObject(self.cfg.table)

        # add objects to the scene
        self.scene.rigid_objects["table"] = self.table
        self.scene.rigid_objects["pillar1"] = self.pillar1
        self.scene.rigid_objects["pillar2"] = self.pillar2
        self.scene.articulations["left_robot"] = self.left_robot
        self.scene.articulations["right_robot"] = self.right_robot

    
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        self.robots=[self.left_robot,self.right_robot]
        self.scene.filter_collisions()

    def _pre_physics_step(self, actions: torch.Tensor) -> None:

        if actions.shape[1] != 14:
            raise ValueError(f"Expected actions dimension 14, got {actions.shape[1]}")

        # self.decimation_step=0
        self.actions = self.action_scale * actions.clone()   # [num_envs, 14]
        # self.step_count = self.step_count + 1

        q_des = self.action_offset  + self.action_scale * actions.clamp(-1.0, 1.0)
        q_des_left = q_des[:, :7]  # shape: [num_robots, num_actions], num_actions is the dimension of the action space of the controller
        q_des_right = q_des[:, 7:]

        # set controller command
        self.left_controller.set_command(q_des_left)     
        self.right_controller.set_command(q_des_right)

        # update reference index and next frame

        # Get lengths of currently active trajectories
        current_lengths = self.motion_lengths_tensor[self.current_trajectory_idx]  # the length of trajector of each env
        # Advance frame index, looping within the current trajectory
        self.current_frame_idx = torch.remainder(self.current_frame_idx + 1, current_lengths)

        # Fetch the next reference q and qd using the updated indices
        # (Using a loop for clarity when indexing into a list of tensors)
        for i in range(self.num_envs):
            traj_idx = self.current_trajectory_idx[i].item() # Use .item() for indexing Python list
            frame_idx = self.current_frame_idx[i].item()
            self.ref_q_next[i] = self.expert_trajectories[traj_idx][frame_idx, :14]
            self.ref_qd_next[i] = self.expert_trajectories[traj_idx][frame_idx, 14:]
            
    def _apply_action(self) -> None:

        left_tau = self.left_controller.compute(
            dof_pos=self.left_robot.data.joint_pos[:, self.arm_joint_ids],
            dof_vel=self.left_robot.data.joint_vel[:, self.arm_joint_ids],
            mass_matrix=self.left_robot.root_physx_view.get_generalized_mass_matrices()[:, self.arm_joint_ids, :][:, :, self.arm_joint_ids],
            gravity=self.left_robot.root_physx_view.get_gravity_compensation_forces()[:, self.arm_joint_ids]
        )
        
        right_tau = self.right_controller.compute(
            dof_pos=self.right_robot.data.joint_pos[:, self.arm_joint_ids],
            dof_vel=self.right_robot.data.joint_vel[:, self.arm_joint_ids],
            mass_matrix=self.right_robot.root_physx_view.get_generalized_mass_matrices()[:, self.arm_joint_ids, :][:, :, self.arm_joint_ids],
            gravity=self.right_robot.root_physx_view.get_gravity_compensation_forces()[:, self.arm_joint_ids]
        )

        self.left_robot.set_joint_effort_target(left_tau, joint_ids=self.arm_joint_ids)
        self.right_robot.set_joint_effort_target(right_tau, joint_ids=self.arm_joint_ids)    
  

    def _get_observations(self) -> dict:

        self._update_internal_joint_states()

        # --- policy obs ---
        # s_t^p
        q = self.joint_pos
        qd = self.joint_vel
      
        # s_t^{g-mimic}
        q_ref = self.ref_q_next
        qd_ref = self.ref_qd_next

        q_delta = q_ref - q
        qd_delta = qd_ref - qd

        obs = torch.cat([
            q,   #  [num_envs, 14]
            qd,  #  [num_envs, 14]
            q_delta,
            qd_delta,
            q_ref,
            qd_ref
        ], dim=-1)  
        observations = {"policy": obs}

        # --- amp obs ---
        amp_obs_step = torch.cat([q, qd], dim=-1)
        # update AMP observation history
        self.amp_observation_buffer[:, 1:] = self.amp_observation_buffer[:, :-1].clone()
        # build AMP observation
        self.amp_observation_buffer[:, 0] = amp_obs_step

        self.extras["amp_obs"] = self.amp_observation_buffer.view(self.num_envs, -1)  #

        return observations

    def _get_rewards(self) -> torch.Tensor:

        self._update_internal_joint_states()

        task_reward = compute_task_rewards(
            q=self.joint_pos,
            qd=self.joint_vel,
            q_ref=self.ref_q_next,
            qd_ref=self.ref_qd_next,
            joint_pos_distance_scale=self.cfg.joint_pos_distance_scale,
            joint_vel_distance_scale=self.cfg.joint_vel_distance_scale
        )
        # amp_obs = self._get_observations()["policy"]
        # amp_reward = self.get_amp_reward(amp_obs)

        # total_reward = self.cfg.task_reward_scale * task_reward + self.cfg.style_reward_scale * amp_reward
    
        return task_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # off_table =  torch.logical_and(self.box.data.root_pos_w[:, 2] < 0.7,self.episode_length_buf>=2)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = torch.zeros_like(time_out)
        return terminated, time_out

    
    def _reset_idx(self, env_ids: Sequence[int] | None):
        """
        Args: 
            env_ids: the env list to reset
        """
        if env_ids is None:
            env_ids = self.left_robot._ALL_INDICES
        num_resets = len(env_ids)
        # reset state to expert trajectory 
        # Randomly select a trajectory for each reset env
        reset_trajectory_ids = torch.randint(0, self.num_trajectories, (num_resets,), device=self.device)
        self.current_trajectory_idx[env_ids] = reset_trajectory_ids

        # Randomly select a starting frame within the chosen trajectory
        selected_lengths = self.motion_lengths_tensor[reset_trajectory_ids]
        # Ensure we don't sample from length 0 trajectories if they exist
        valid_lengths = torch.clamp(selected_lengths, min=1)
        reset_frame_ids = (torch.rand(num_resets, device=self.device) * valid_lengths).long() # Sample in [0, L-1]
        self.current_frame_idx[env_ids] = reset_frame_ids

        # Fetch initial state (q, qd) from the sampled expert state
        # (Using loop for clarity with list of tensors)
        initial_q = torch.zeros((num_resets, 14), device=self.device)
        initial_qd = torch.zeros((num_resets, 14), device=self.device)
        for i in range(num_resets):
            traj_idx = reset_trajectory_ids[i].item()
            frame_idx = reset_frame_ids[i].item()
            initial_q[i] = self.expert_trajectories[traj_idx][frame_idx, :14]
            initial_qd[i] = self.expert_trajectories[traj_idx][frame_idx, 14:]
        # later can add noise to initial state here if needed

        # Write initial state to simulation
        self.left_robot.write_joint_state_to_sim(initial_q[:, :7], initial_qd[:, :7], None, env_ids)
        self.right_robot.write_joint_state_to_sim(initial_q[:, 7:], initial_qd[:, 7:], None, env_ids)
        # Update internal state buffers
        self.joint_pos[env_ids] = initial_q
        self.joint_vel[env_ids] = initial_qd

        # Set the *next* reference frame for the first observation calculation
        next_frame_ids = torch.remainder(reset_frame_ids + 1, selected_lengths)
        for i in range(num_resets):
            # Need env_id corresponding to loop index i
            current_env_id = env_ids[i]
            traj_idx = reset_trajectory_ids[i].item()
            frame_idx = next_frame_ids[i].item()
            self.ref_q_next[current_env_id] = self.expert_trajectories[traj_idx][frame_idx, :14]
            self.ref_qd_next[current_env_id] = self.expert_trajectories[traj_idx][frame_idx, 14:]

        # initialize AMP observation buffer 
        # Collect reference history corresponding to the reset state
        initial_amp_history = self.collect_reference_motions(num_resets, reset_trajectory_ids, reset_frame_ids)
        # Fill the buffer for the reset environments
        self.amp_observation_buffer[env_ids] = initial_amp_history.view(
            num_resets, self.cfg.num_amp_observations, self.single_amp_observation_space
        )
  
        # Call parent reset AFTER setting states and buffers
        super()._reset_idx(env_ids)  
    
    def collect_reference_motions(self, num_samples: int,
                                trajectory_indices: torch.Tensor | None = None,
                                current_frame_indices: torch.Tensor | None = None) -> torch.Tensor:
        """

        Args:
            num_samples: Number of sequences to generate.
            trajectory_indices: Trajectory index for each sample (shape: [num_samples]). Random if None.
            current_frame_indices: Starting frame index within trajectory for each sample (shape: [num_samples]). Random if None.

        Returns:
            Tensor containing reference sequences, shape (num_samples, self.amp_observation_size).
        """
        # --- Input Validation and Default Sampling ---
        if trajectory_indices is None:
            trajectory_indices = torch.randint(0, self.num_trajectories, (num_samples,), device=self.device)
        if current_frame_indices is None:
            selected_lengths = self.motion_lengths_tensor[trajectory_indices]
            valid_lengths = torch.clamp(selected_lengths, min=1) # Avoid division by zero or modulo zero
            current_frame_indices = (torch.rand(num_samples, device=self.device) * valid_lengths).long()

        if trajectory_indices.shape != (num_samples,) or current_frame_indices.shape != (num_samples,):
            raise ValueError(f"Indices tensors must have shape ({num_samples},)")

        # Sequence of steps back in time: [0, 1, ..., N-1] where N = num_amp_observations
        time_steps = torch.arange(self.cfg.num_amp_observations, device=self.device)
        # Calculate indices relative to the current frame: [current, current-1, ..., current-N+1]
        # Shape: [num_samples, num_amp_observations]
        relative_indices = current_frame_indices.unsqueeze(1) - time_steps.unsqueeze(0)
        # Clamp indices to be non-negative (cannot sample before trajectory start)
        history_indices = torch.clamp(relative_indices, min=0)

        # Vectorizing sampling from a list of tensors with varying indices is complex.
        # A loop is clearer and often acceptable unless this becomes a major bottleneck.
        all_sequences = []
        for i in range(num_samples):
            traj_idx = trajectory_indices[i].item()
            # Indices for this specific sample's history (shape: [num_amp_observations])
            hist_idx_sample = history_indices[i]
            # Sample data from the correct trajectory tensor using the historical indices
            # sequence_data shape: [num_amp_observations, 28]
            sequence_data = self.expert_trajectories[traj_idx][hist_idx_sample]
            all_sequences.append(sequence_data)

        # Stack sequences from all samples
        # amp_obs_sequences shape: [num_samples, num_amp_observations, 28]
        if not all_sequences: # Handle edge case if num_samples was 0
             return torch.empty((0, self.amp_observation_size), device=self.device)
        amp_obs_sequences = torch.stack(all_sequences, dim=0)

        # Flatten the history dimension to match discriminator input requirement
        # final_output shape: [num_samples, num_amp_observations * 28]
        final_output = amp_obs_sequences.view(num_samples, -1)
 

        return final_output
    

    # def get_amp_reward(self, obs:torch.Tensor) -> torch.Tensor:
    #     with torch.no_grad():
    #         amp_prob, _ = self.amp_discriminator(obs)
    #         amp_reward = -torch.log(1.0 - amp_prob + 1e-6)
    #         amp_reward = amp_reward.squeeze(-1)
    #     return amp_reward

    def _update_internal_joint_states(self):
        """Read current joint data from sim buffers and updates internal state variables."""
        joint_pos_left = self.left_robot.data.joint_pos
        joint_vel_left = self.left_robot.data.joint_vel
        joint_pos_right = self.right_robot.data.joint_pos
        joint_vel_right = self.right_robot.data.joint_vel

        self.joint_pos = torch.cat([joint_pos_left, joint_pos_right], dim=1)
        self.joint_vel = torch.cat([joint_vel_left, joint_vel_right], dim=1)

        

@torch.jit.script
def compute_task_rewards(
    q: torch.Tensor,
    qd: torch.Tensor,
    q_ref: torch.Tensor,
    qd_ref: torch.Tensor,
    joint_pos_distance_scale: float,
    joint_vel_distance_scale: float
    ) -> torch.Tensor:

    # Reward calculations

    q_diff = q_ref - q
    qd_diff = qd_ref - qd 

    q_norm = torch.norm(q_diff, dim=1)
    qd_norm = torch.norm(qd_diff, dim=1)

    pos_reward = joint_pos_distance_scale * torch.exp(-10.0 * q_norm)
    vel_reward = joint_vel_distance_scale * torch.exp(-0.1 * qd_norm)

    # Total reward  
    total_reward = pos_reward + vel_reward
    
    return total_reward




