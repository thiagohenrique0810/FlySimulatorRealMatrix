"""Multiple flies walking together in the MuJoCo viewer, with inter-fly collisions."""

import itertools
import numpy as np
import mujoco.viewer

from flygym.compose import Fly, KinematicPosePreset, ActuatorType, FlatGroundWorld
from flygym.compose.physics import ContactParams
from flygym.anatomy import (
    Skeleton, AxisOrder, JointPreset, ActuatedDOFPreset, ContactBodiesPreset,
)
from flygym.utils.math import Rotation3D
from flygym import Simulation
from flygym_demo.spotlight_data import MotionSnippet

# --- Config ---
NUM_FLIES = 3
SPACING = 2.0  # mm between flies (closer so they can bump into each other)

# --- Load experimental walking data ---
snippet = MotionSnippet()

# --- Build multiple flies ---
axis_order = AxisOrder.YAW_PITCH_ROLL
neutral_pose = KinematicPosePreset.NEUTRAL
actuator_type = ActuatorType.POSITION
actuator_gain = 150.0

flies = []
for i in range(NUM_FLIES):
    fly = Fly(name=f"fly{i}")
    skeleton = Skeleton(axis_order=axis_order, joint_preset=JointPreset.LEGS_ONLY)
    fly.add_joints(skeleton, neutral_pose=neutral_pose)
    actuated_dofs = skeleton.get_actuated_dofs_from_preset(ActuatedDOFPreset.LEGS_ACTIVE_ONLY)
    fly.add_actuators(
        actuated_dofs,
        actuator_type=actuator_type,
        kp=actuator_gain,
        neutral_input=neutral_pose,
    )
    fly.add_leg_adhesion()
    fly.colorize()
    flies.append(fly)

# Add tracking camera to the first fly
flies[0].add_tracking_camera()

# --- Build world with all flies ---
world = FlatGroundWorld()
for i, fly in enumerate(flies):
    y_offset = (i - (NUM_FLIES - 1) / 2) * SPACING
    world.add_fly(
        fly,
        [0, y_offset, 0.7],
        Rotation3D(format="quat", values=[1, 0, 0, 0]),
    )

# --- Add inter-fly contact pairs ---
# Get the body segments used for collision
contact_bodysegs = ContactBodiesPreset.LEGS_THORAX_ABDOMEN_HEAD.to_body_segments_list()
contact_params = ContactParams()

for fly_a, fly_b in itertools.combinations(flies, 2):
    for seg_a in contact_bodysegs:
        for seg_b in contact_bodysegs:
            geom_a = fly_a.mjcf_root.find("geom", seg_a.name)
            geom_b = fly_b.mjcf_root.find("geom", seg_b.name)
            if geom_a is not None and geom_b is not None:
                world.mjcf_root.contact.add(
                    "pair",
                    geom1=geom_a,
                    geom2=geom_b,
                    name=f"{fly_a.name}_{seg_a.name}__{fly_b.name}_{seg_b.name}",
                    friction=contact_params.get_friction_tuple(),
                    solref=contact_params.get_solref_tuple(),
                    solimp=contact_params.get_solimp_tuple(),
                    margin=contact_params.margin,
                )

# --- Prepare simulation ---
sim_timestep = 1e-4
sim = Simulation(world)

# Get joint angle targets for each fly (with slight phase offset)
joint_angles_per_fly = {}
for i, fly in enumerate(flies):
    joint_angles_per_fly[fly.name] = snippet.get_joint_angles(
        output_timestep=sim_timestep,
        output_dof_order=fly.get_actuated_jointdofs_order(actuator_type),
    )

sim_duration = snippet.joint_angles.shape[0] / snippet.data_fps
nsteps = int(sim_duration / sim_timestep)

# Phase offsets so they don't walk in perfect sync (in steps)
phase_offsets = [0, nsteps // 3, 2 * nsteps // 3]

# --- Launch viewer ---
sim.reset()
for fly in flies:
    sim.set_leg_adhesion_states(fly.name, np.ones(6, dtype=np.bool_))

print(f"{NUM_FLIES} flies walking — close the viewer window to stop.")

with mujoco.viewer.launch_passive(sim.mj_model, sim.mj_data) as viewer:
    step_idx = 0
    while viewer.is_running():
        for i, fly in enumerate(flies):
            idx = (step_idx + phase_offsets[i]) % nsteps
            target = joint_angles_per_fly[fly.name][idx, :]
            sim.set_actuator_inputs(fly.name, actuator_type, target)

        sim.step()
        viewer.sync()
        step_idx += 1
