"""Make the fly walk with live visualization in the MuJoCo viewer."""

import time
import numpy as np
import mujoco.viewer

from flygym.compose import Fly, KinematicPosePreset, ActuatorType, FlatGroundWorld
from flygym.anatomy import Skeleton, AxisOrder, JointPreset, ActuatedDOFPreset
from flygym.utils.math import Rotation3D
from flygym import Simulation
from flygym_demo.spotlight_data import MotionSnippet

# --- Load experimental walking data ---
snippet = MotionSnippet()
print(f"Loaded: {snippet.joint_angles.shape[0]} frames at {snippet.data_fps} Hz")

# --- Build fly model ---
axis_order = AxisOrder.YAW_PITCH_ROLL
neutral_pose = KinematicPosePreset.NEUTRAL
actuator_type = ActuatorType.POSITION
actuator_gain = 150.0

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
fly.add_tracking_camera()

# --- Build world & simulation ---
world = FlatGroundWorld()
world.add_fly(fly, [0, 0, 0.7], Rotation3D(format="quat", values=[1, 0, 0, 0]))

sim_timestep = 1e-4
sim = Simulation(world)

# Prepare joint angle targets
joint_angles_nmf = snippet.get_joint_angles(
    output_timestep=sim_timestep,
    output_dof_order=fly.get_actuated_jointdofs_order(actuator_type),
)

sim_duration = snippet.joint_angles.shape[0] / snippet.data_fps
nsteps = int(sim_duration / sim_timestep)

# --- Launch passive viewer using sim's own mj_model/mj_data ---
fly_name = fly.name
sim.reset()
sim.set_leg_adhesion_states(fly_name, np.ones(6, dtype=np.bool_))

print(f"Launching viewer — simulating {sim_duration:.1f}s of walking (loops forever)")
print("Close the viewer window to stop.")

with mujoco.viewer.launch_passive(sim.mj_model, sim.mj_data) as viewer:
    step_idx = 0
    while viewer.is_running():
        # Set target angles (loop the motion)
        idx = step_idx % nsteps
        target_angles = joint_angles_nmf[idx, :]
        sim.set_actuator_inputs(fly_name, actuator_type, target_angles)

        # Step physics
        sim.step()

        # Sync viewer at roughly real-time
        viewer.sync()
        step_idx += 1
