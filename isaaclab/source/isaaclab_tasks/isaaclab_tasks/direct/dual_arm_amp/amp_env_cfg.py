# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import torch
from collections.abc import Sequence
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg
from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
import isaaclab.envs.mdp as mdp
import random
from franka_panda import LEFT_ROBOT_CFG, RIGHT_ROBOT_CFG

from isaaclab.managers import SceneEntityCfg

NUM_ENVS = 10


@configclass
class EventCfg:
    """Configuration for randomization."""

    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("table"),
            "static_friction_range": (0.5, 2.0),
            "dynamic_friction_range": (0.5, 2.0),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )


    
@configclass
class AmpMotionImitatorEnvCfg(DirectRLEnvCfg):
    # env
    #decimation = 5
    #episode_length_s = 8.0
    # action_scale = 1.5
    action_space = 14
    observation_space = 84
    state_space =0
    num_amp_observations = 10
    motion_files = [
        "/home/nikki/data1/ex1_trajectory1.pt",
        "/home/nikki/data1/ex1_trajectory2.pt",
        "/home/nikki/data1/ex2_trajectory1.pt",
        "/home/nikki/data1/ex2_trajectory2.pt",
        "/home/nikki/data1/ex3_trajectory1.pt",
        "/home/nikki/data1/ex3_trajectory2.pt",
        "/home/nikki/data1/ex4_trajectory1.pt",
        "/home/nikki/data1/ex4_trajectory2.pt",
        "/home/nikki/data1/ex6_trajectory1.pt",
        "/home/nikki/data1/ex6_trajectory2.pt",
        "/home/nikki/data1/ex6_trajectory3.pt",
        "/home/nikki/data1/ex6_trajectory4.pt",
    ]

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 100, render_interval=decimation)   # 100 Hz

    # events
    events: EventCfg = EventCfg()

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # table
    table = RigidObjectCfg(
        prim_path="/World/envs/.*/table",
        spawn=sim_utils.CuboidCfg(
            size=(0.88, 1.88, 1.6),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            activate_contact_sensors=True,
            mass_props=sim_utils.MassPropertiesCfg(mass=100.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.2, 0.7, 0.6),
                opacity=1.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.6, 0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # pillars to support robot
    pillar1= RigidObjectCfg(
        prim_path="/World/envs/.*/pillar1",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.8),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=100.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.2, 0.7, 0.6),
                opacity=1.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0, -0.445, 0.4),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    pillar2= RigidObjectCfg(
        prim_path="/World/envs/.*/pillar2",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.8),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            activate_contact_sensors=True,
            mass_props=sim_utils.MassPropertiesCfg(mass=100.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.2, 0.7, 0.6),
                opacity=1.0,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0, 0.445, 0.4),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )


    # robots
    left_robot = LEFT_ROBOT_CFG.replace(prim_path="/World/envs/.*/left_robot")        
    left_robot.spawn.rigid_props.disable_gravity = False

    right_robot = RIGHT_ROBOT_CFG.replace(prim_path="/World/envs/.*/right_robot")
    right_robot.spawn.rigid_props.disable_gravity = False


    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=NUM_ENVS, 
        env_spacing=4.0, 
        replicate_physics=False
    )
            



    # reward scales
    joint_pos_distance_scale = 1.0
    joint_vel_distance_scale = 1.0
    # task_reward_scale = 1.0
    # style_reward_scale = 0.2
    # gradient_penalty_scale = 0.5
