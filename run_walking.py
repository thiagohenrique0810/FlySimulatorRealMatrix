"""Make the fly walk by replaying experimentally recorded joint angles."""

import numpy as np
from tqdm import trange

from flygym.compose import Fly, KinematicPosePreset, ActuatorType, FlatGroundWorld
from flygym.anatomy import Skeleton, AxisOrder, JointPreset, ActuatedDOFPreset
from flygym.utils.math import Rotation3D
from flygym import Simulation
from flygym_demo.spotlight_data import MotionSnippet

# --- Load experimental walking data ---
snippet = MotionSnippet()
print(f"Loaded motion snippet: {snippet.joint_angles.shape[0]} frames at {snippet.data_fps} Hz")

# --- Build fly model ---
axis_order = AxisOrder.YAW_PITCH_ROLL
neutral_pose = KinematicPosePreset.NEUTRAL
actuator_type = ActuatorType.POSITION
actuator_gain = 150.0  # torque per angular error (uN*mm/rad)

fly = Fly()
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
tracking_cam = fly.add_tracking_camera()

# --- Build world ---
world = FlatGroundWorld()
world.add_fly(
    fly,
    [0, 0, 0.7],
    Rotation3D(format="quat", values=[1, 0, 0, 0]),
)

# --- Prepare simulation ---
sim_timestep = 1e-4
sim = Simulation(world)
sim.set_renderer(tracking_cam)

# Convert experimental joint angles to simulation timestep
joint_angles_nmf = snippet.get_joint_angles(
    output_timestep=sim_timestep,
    output_dof_order=fly.get_actuated_jointdofs_order(actuator_type),
)

sim_duration = snippet.joint_angles.shape[0] / snippet.data_fps
nsteps = int(sim_duration / sim_timestep)
print(f"Simulating {sim_duration:.2f}s ({nsteps} steps)...")

# --- Run simulation ---
fly_name = fly.name
sim.reset()
sim.set_leg_adhesion_states(fly_name, np.ones(6, dtype=np.bool_))

for step_idx in trange(nsteps, desc="Walking simulation"):
    target_angles = joint_angles_nmf[step_idx, :]
    sim.set_actuator_inputs(fly_name, actuator_type, target_angles)
    sim.step()
    sim.render_as_needed()

# --- Save video ---
output_path = "fly_walking.mp4"
sim.renderer.save_video(output_path)
print(f"Video saved to {output_path}")
