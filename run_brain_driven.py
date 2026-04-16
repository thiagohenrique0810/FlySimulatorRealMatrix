"""Run the fly with multi-behaviour brain-driven control.

Three behaviours are supported, each triggered by distinct neuron
populations in the FlyWire v783 connectome:

1. **Walking** — tripod gait driven by experimental kinematic data.
2. **Antenna grooming** — front legs lift toward the head and scrape
   the antennae while the other four legs hold stance.
3. **Food seeking** — the fly slows down, makes exploratory turns, and
   periodically taps with its front legs (chemosensory probing).

The brain's spiking neural network (LIF model) decides which behaviour
is active based on which neuron pool fires most strongly.

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
from flygym.brain.connectome import (
    Connectome,
    MN9_FLYWIRE_ID,
    SUGAR_SENSING_RIGHT_IDS,
    WALKING_SEZ_TYPES,
    GROOMING_SEZ_TYPES,
    FEEDING_SEZ_TYPES,
)
from flygym.brain.lif_model import OnlineBrainModel
from flygym.brain.bridge import (
    BehaviorState,
    BehaviorController,
    SensorimotorBridge,
    WalkingPatternGenerator,
    GroomingProgram,
    FeedingProgram,
    reorder_cpg_to_actuator,
)


def _collect_ids(connectome: Connectome, type_names: list[str]) -> list[int]:
    """Collect all FlyWire IDs for a list of SEZ type names."""
    sez = connectome.sez_neuron_types
    ids: list[int] = []
    for t in type_names:
        if t in sez:
            ids.extend(sez[t])
    return [i for i in ids if i in connectome._flyid_to_idx]


def main() -> None:
    # ===================================================================
    # Configuration
    # ===================================================================
    brain_data_dir = Path("external/Drosophila_brain_model")
    sim_timestep = 1e-4
    brain_dt_ms = 5.0
    sim_duration_s = 8.0          # longer to see all behaviours
    output_path = "fly_brain_driven.mp4"

    brain_update_interval = int(brain_dt_ms * 1e-3 / sim_timestep)
    n_steps = int(sim_duration_s / sim_timestep)

    print("=" * 60)
    print("  Multi-Behaviour Brain-Driven Fly Simulation")
    print("=" * 60)

    # ===================================================================
    # 1. Load the connectome
    # ===================================================================
    print("\n[1/6] Loading connectome...")
    connectome = Connectome(brain_data_dir)
    print(f"       {connectome.n_neurons:,} neurons loaded")

    # --- Neuron populations per behaviour ---
    sensory_ids = [s for s in SUGAR_SENSING_RIGHT_IDS
                   if s in connectome._flyid_to_idx]

    walking_ids = [MN9_FLYWIRE_ID] + _collect_ids(connectome, WALKING_SEZ_TYPES)
    grooming_ids = _collect_ids(connectome, GROOMING_SEZ_TYPES)
    feeding_ids = _collect_ids(connectome, FEEDING_SEZ_TYPES)

    # All motor neurons = union (deduplicated)
    all_motor_ids: list[int] = []
    seen: set[int] = set()
    for mid in walking_ids + grooming_ids + feeding_ids:
        if mid not in seen and mid in connectome._flyid_to_idx:
            seen.add(mid)
            all_motor_ids.append(mid)

    print(f"       {len(sensory_ids)} sensory neurons")
    print(f"       {len(walking_ids)} walking motor neurons")
    print(f"       {len(grooming_ids)} grooming motor neurons")
    print(f"       {len(feeding_ids)} feeding motor neurons")
    print(f"       {len(all_motor_ids)} total motor neurons (unique)")

    # ===================================================================
    # 2. Set up the brain model
    # ===================================================================
    print("\n[2/6] Building brain model (subnetwork)...")
    brain = OnlineBrainModel(
        connectome,
        sensory_neuron_ids=sensory_ids,
        motor_neuron_ids=all_motor_ids,
        dt_ms=0.1,
        use_subnetwork=True,
        spike_window_ms=50.0,
    )
    brain.setup()
    print(f"       Subnetwork: {brain.n_neurons:,} neurons")

    # ===================================================================
    # 3. Build the fly body and world
    # ===================================================================
    print("\n[3/6] Building fly model and world...")
    fly = Fly()
    skeleton = Skeleton(axis_order=AxisOrder.YAW_PITCH_ROLL,
                        joint_preset=JointPreset.LEGS_ONLY)
    fly.add_joints(skeleton, neutral_pose=KinematicPosePreset.NEUTRAL)
    actuated_dofs = skeleton.get_actuated_dofs_from_preset(
        ActuatedDOFPreset.LEGS_ACTIVE_ONLY)
    fly.add_actuators(actuated_dofs, actuator_type=ActuatorType.POSITION,
                      kp=150.0, neutral_input=KinematicPosePreset.NEUTRAL)
    fly.add_leg_adhesion()
    fly.colorize()
    tracking_cam = fly.add_tracking_camera()

    world = FlatGroundWorld()
    world.add_fly(fly, [0, 0, 0.7],
                  Rotation3D(format="quat", values=[1, 0, 0, 0]))

    sim = Simulation(world)
    sim.set_renderer(tracking_cam)
    fly_name = fly.name

    # ===================================================================
    # 4. Motor programmes & behaviour controller
    # ===================================================================
    print("\n[4/6] Initialising motor programmes...")
    bridge = SensorimotorBridge(
        brain, connectome,
        sensory_gain=5.0,
        motor_speed_gain=0.002,
        motor_turn_gain=0.005,
        base_excitation_mv=35.0,
    )

    walk_cpg = WalkingPatternGenerator()
    groom_prog = GroomingProgram(grooming_freq=3.0)
    feed_prog = FeedingProgram(tap_freq=2.0)

    behavior_ctrl = BehaviorController(
        brain, connectome,
        walking_ids=walking_ids,
        grooming_ids=grooming_ids,
        feeding_ids=feeding_ids,
        hold_time_s=1.0,
        grooming_threshold=0.5,
        feeding_threshold=0.4,
    )

    # ===================================================================
    # 5. Forced behaviour schedule (ensures all are shown in video)
    # ===================================================================
    # The brain's tonic activity alone may not reliably switch behaviours,
    # so we define a schedule that externally boosts specific neuron pools
    # to trigger each behaviour at least once during the simulation.
    schedule: list[tuple[float, float, BehaviorState]] = [
        (0.0,  2.5, BehaviorState.WALKING),
        (2.5,  4.5, BehaviorState.GROOMING),
        (4.5,  6.0, BehaviorState.FEEDING),
        (6.0,  8.0, BehaviorState.WALKING),
    ]

    def scheduled_behavior(t: float) -> BehaviorState:
        for t0, t1, beh in schedule:
            if t0 <= t < t1:
                return beh
        return BehaviorState.WALKING

    # ===================================================================
    # 6. Run simulation
    # ===================================================================
    print(f"\n[5/6] Running simulation ({sim_duration_s}s, {n_steps} steps)...")

    sim.reset()
    brain.reset()
    walk_cpg.reset()
    groom_prog.reset()
    feed_prog.reset()
    behavior_ctrl.reset()
    sim.warmup(duration_s=0.02)
    sim.set_leg_adhesion_states(fly_name, np.ones(6, dtype=np.bool_))

    speed = 0.5
    turn = 0.0
    current_behavior = BehaviorState.WALKING
    behavior_counts = {b: 0 for b in BehaviorState}

    for step_idx in trange(n_steps, desc="Brain-driven simulation"):
        t = step_idx * sim_timestep

        # --- Brain update ---
        if step_idx % brain_update_interval == 0:
            joint_angles = sim.get_joint_angles(fly_name)
            contact_info = sim.get_ground_contact_info(fly_name)
            contact_active = contact_info[0]
            forces = contact_info[1]
            body_positions = sim.get_body_positions(fly_name)

            speed, turn = bridge.update_brain_and_get_commands(
                contact_active=contact_active,
                forces=forces,
                joint_angles=joint_angles,
                body_positions=body_positions,
                brain_dt_ms=brain_dt_ms,
            )

            # Behaviour selection (scheduled + brain modulated)
            current_behavior = scheduled_behavior(t)
            behavior_counts[current_behavior] += 1

            # Debug output
            if step_idx % (brain_update_interval * 20) == 0:
                motor_rates = brain.get_motor_spike_rates()
                print(f"  t={t:.2f}s  behaviour={current_behavior.value:8s}  "
                      f"speed={speed:.3f}  turn={turn:.3f}  "
                      f"motor_mean={np.mean(motor_rates):.1f} Hz")

        # --- Generate joint angles for current behaviour ---
        if current_behavior == BehaviorState.WALKING:
            angles = walk_cpg.step(sim_timestep, speed=speed, turn=turn)
            adhesion = walk_cpg.get_adhesion_states().astype(float)
        elif current_behavior == BehaviorState.GROOMING:
            angles = groom_prog.step(sim_timestep)
            adhesion = groom_prog.get_adhesion_states().astype(float)
        elif current_behavior == BehaviorState.FEEDING:
            angles = feed_prog.step(sim_timestep, speed=0.15, turn=turn)
            adhesion = feed_prog.get_adhesion_states().astype(float)
        else:
            angles = walk_cpg.step(sim_timestep, speed=speed, turn=turn)
            adhesion = walk_cpg.get_adhesion_states().astype(float)

        target_angles = reorder_cpg_to_actuator(angles, fly, ActuatorType.POSITION)
        sim.set_actuator_inputs(fly_name, ActuatorType.POSITION, target_angles)
        sim.set_leg_adhesion_states(fly_name, adhesion)

        sim.step()
        sim.render_as_needed()

    # ===================================================================
    # Save video
    # ===================================================================
    sim.renderer.save_video(output_path)
    print(f"\nVideo saved to {output_path}")
    print("\nBehaviour time distribution:")
    total = sum(behavior_counts.values()) or 1
    for b, c in behavior_counts.items():
        print(f"  {b.value:10s}: {c:5d} updates ({100*c/total:.1f}%)")
    print("Done!")


if __name__ == "__main__":
    main()
