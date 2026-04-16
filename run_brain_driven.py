"""Run the fly with its body controlled by the connectome-based brain model.

This script integrates the Drosophila whole-brain LIF model (Shiu et al.)
with the FlyGym physics simulation.  The closed loop works as follows:

    1. **Sense** — read ground contacts, joint angles, and body positions
       from the physics simulation.
    2. **Brain step** — inject sensory currents into the brain model, advance
       the spiking network, and read out motor neuron firing rates.
    3. **Act** — convert firing rates into walking speed and turning
       commands, generate joint-angle targets with a CPG, and send them to
       the simulation actuators.

The brain model uses a subnetwork of the FlyWire v783 connectome centred on
SEZ motor-relevant neurons to keep real-time performance practical.

Usage:
    python run_brain_driven.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from tqdm import trange

# ---------------------------------------------------------------------------
# FlyGym imports
# ---------------------------------------------------------------------------
from flygym.compose import Fly, KinematicPosePreset, ActuatorType, FlatGroundWorld
from flygym.anatomy import Skeleton, AxisOrder, JointPreset, ActuatedDOFPreset
from flygym.utils.math import Rotation3D
from flygym import Simulation

# ---------------------------------------------------------------------------
# Brain model imports
# ---------------------------------------------------------------------------
from flygym.brain.connectome import Connectome, MN9_FLYWIRE_ID, SUGAR_SENSING_RIGHT_IDS
from flygym.brain.lif_model import OnlineBrainModel
from flygym.brain.bridge import (
    SensorimotorBridge,
    WalkingPatternGenerator,
    reorder_cpg_to_actuator,
)


def main() -> None:
    # ===================================================================
    # Configuration
    # ===================================================================
    brain_data_dir = Path("external/Drosophila_brain_model")
    sim_timestep = 1e-4           # Physics timestep (s)
    brain_dt_ms = 5.0             # Brain model update interval (ms)
    sim_duration_s = 5.0          # Total simulation time (s)
    output_path = "fly_brain_driven.mp4"

    # How many physics steps between brain updates
    brain_update_interval = int(brain_dt_ms * 1e-3 / sim_timestep)  # 50

    # Total steps
    n_steps = int(sim_duration_s / sim_timestep)

    print("=" * 60)
    print("  Brain-Driven Fly Simulation")
    print("=" * 60)

    # ===================================================================
    # 1. Load the connectome
    # ===================================================================
    print("\n[1/5] Loading connectome...")
    connectome = Connectome(brain_data_dir)
    print(f"       {connectome.n_neurons:,} neurons loaded")
    print(f"       {len(connectome.sez_neuron_types)} SEZ neuron types")

    # --- Define sensory and motor neuron populations ---
    # Sensory: use the sugar-sensing neurons as a representative sensory input
    # Filter to IDs that actually exist in this connectome snapshot
    sensory_ids = [sid for sid in SUGAR_SENSING_RIGHT_IDS if sid in connectome._flyid_to_idx]

    # Motor: use SEZ neurons that are likely motor-related
    # Pick a diverse subset of SEZ types + known MN9
    motor_ids = [MN9_FLYWIRE_ID]
    sez_types = connectome.sez_neuron_types
    motor_type_names = [
        "rocket", "basket", "horseshoe", "diatom", "mime",
        "weaver", "gallinule", "mandala", "brontosaraus", "oink",
    ]
    for tname in motor_type_names:
        if tname in sez_types:
            motor_ids.extend(sez_types[tname])

    # Deduplicate while preserving order
    seen: set[int] = set()
    unique_motor_ids: list[int] = []
    for mid in motor_ids:
        if mid not in seen and mid in connectome._flyid_to_idx:
            seen.add(mid)
            unique_motor_ids.append(mid)
    motor_ids = unique_motor_ids

    print(f"       {len(sensory_ids)} sensory neurons")
    print(f"       {len(motor_ids)} motor neurons")

    # ===================================================================
    # 2. Set up the brain model
    # ===================================================================
    print("\n[2/5] Building brain model (subnetwork)...")
    brain = OnlineBrainModel(
        connectome,
        sensory_neuron_ids=sensory_ids,
        motor_neuron_ids=motor_ids,
        dt_ms=0.1,
        use_subnetwork=True,    # use local subnetwork for speed
        spike_window_ms=50.0,
    )
    brain.setup()
    print(f"       Subnetwork: {brain.n_neurons:,} neurons")
    print(f"       {brain.n_sensory} sensory, {brain.n_motor} motor neurons")

    # ===================================================================
    # 3. Build the fly body and world
    # ===================================================================
    print("\n[3/5] Building fly model and world...")
    axis_order = AxisOrder.YAW_PITCH_ROLL
    neutral_pose = KinematicPosePreset.NEUTRAL
    actuator_type = ActuatorType.POSITION
    actuator_gain = 150.0

    fly = Fly()
    skeleton = Skeleton(axis_order=axis_order, joint_preset=JointPreset.LEGS_ONLY)
    fly.add_joints(skeleton, neutral_pose=neutral_pose)

    actuated_dofs = skeleton.get_actuated_dofs_from_preset(
        ActuatedDOFPreset.LEGS_ACTIVE_ONLY
    )
    fly.add_actuators(
        actuated_dofs,
        actuator_type=actuator_type,
        kp=actuator_gain,
        neutral_input=neutral_pose,
    )
    fly.add_leg_adhesion()
    fly.colorize()
    tracking_cam = fly.add_tracking_camera()

    world = FlatGroundWorld()
    world.add_fly(
        fly,
        [0, 0, 0.7],
        Rotation3D(format="quat", values=[1, 0, 0, 0]),
    )

    sim = Simulation(world)
    sim.set_renderer(tracking_cam)
    fly_name = fly.name

    # ===================================================================
    # 4. Set up the sensorimotor bridge and CPG
    # ===================================================================
    print("\n[4/5] Initialising sensorimotor bridge and CPG...")
    bridge = SensorimotorBridge(
        brain,
        connectome,
        sensory_gain=5.0,
        motor_speed_gain=0.002,
        motor_turn_gain=0.005,
        base_excitation_mv=35.0,
    )
    cpg = WalkingPatternGenerator(base_frequency=12.0)

    # ===================================================================
    # 5. Run the simulation
    # ===================================================================
    print(f"\n[5/5] Running simulation ({sim_duration_s}s, "
          f"{n_steps} steps, brain update every {brain_update_interval} steps)...")

    sim.reset()
    brain.reset()
    cpg.reset()

    # Warmup: let the fly settle on the ground
    sim.warmup(duration_s=0.02)

    # Initial adhesion
    sim.set_leg_adhesion_states(fly_name, np.ones(6, dtype=np.bool_))

    # Tracking variables
    speed = 0.5  # initial walking speed
    turn = 0.0   # initial turning

    for step_idx in trange(n_steps, desc="Brain-driven simulation"):
        # --- Brain update (every brain_update_interval physics steps) ---
        if step_idx % brain_update_interval == 0:
            # Read sensors
            joint_angles = sim.get_joint_angles(fly_name)
            contact_info = sim.get_ground_contact_info(fly_name)
            contact_active = contact_info[0]
            forces = contact_info[1]
            body_positions = sim.get_body_positions(fly_name)

            # Brain cycle: sense → think → act
            speed, turn = bridge.update_brain_and_get_commands(
                contact_active=contact_active,
                forces=forces,
                joint_angles=joint_angles,
                body_positions=body_positions,
                brain_dt_ms=brain_dt_ms,
            )

            # Debug: print brain output periodically
            if step_idx % (brain_update_interval * 20) == 0:
                motor_rates = brain.get_motor_spike_rates()
                print(f"  t={step_idx * sim_timestep:.3f}s  speed={speed:.3f}  "
                      f"turn={turn:.3f}  motor_rate_mean={np.mean(motor_rates):.1f} Hz")

        # --- CPG: generate joint angle targets ---
        cpg_angles = cpg.step(sim_timestep, speed=speed, turn=turn)

        # Reorder to match fly actuator DOF order
        target_angles = reorder_cpg_to_actuator(cpg_angles, fly, actuator_type)

        # Set actuator inputs and adhesion
        sim.set_actuator_inputs(fly_name, actuator_type, target_angles)

        adhesion = cpg.get_adhesion_states().astype(float)
        sim.set_leg_adhesion_states(fly_name, adhesion)

        # Step physics and render
        sim.step()
        sim.render_as_needed()

    # ===================================================================
    # Save video
    # ===================================================================
    sim.renderer.save_video(output_path)
    print(f"\nVideo saved to {output_path}")
    print("Done!")


if __name__ == "__main__":
    main()
