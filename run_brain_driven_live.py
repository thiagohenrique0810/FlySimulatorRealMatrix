"""Run brain-driven fly with live MuJoCo viewer.

The brain now controls:
  1. Behaviour switching — neural activity decides walk/groom/feed
  2. Per-leg stepping speed — each leg's motor neuron pool sets its pace
  3. Joint-level posture modulation — firing variance tweaks stride/height
  4. Responds to simulated sensory events (antenna dust, sugar encounters)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import mujoco.viewer

from flygym.compose import Fly, KinematicPosePreset, ActuatorType
from flygym.brain.nature_world import NatureWorld
from flygym.anatomy import Skeleton, AxisOrder, JointPreset, ActuatedDOFPreset
from flygym.utils.math import Rotation3D
from flygym import Simulation

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
    SensoryEventGenerator,
    SensorimotorBridge,
    WalkingPatternGenerator,
    GroomingProgram,
    FeedingProgram,
    reorder_cpg_to_actuator,
)
from flygym.brain.visualizer import BrainVisualizer


def _collect_ids(connectome: Connectome, type_names: list[str]) -> list[int]:
    sez = connectome.sez_neuron_types
    ids: list[int] = []
    for t in type_names:
        if t in sez:
            ids.extend(sez[t])
    return [i for i in ids if i in connectome._flyid_to_idx]


def main() -> None:
    brain_data_dir = Path("external/Drosophila_brain_model")
    sim_timestep = 1e-4
    brain_dt_ms = 20.0
    brain_update_interval = int(brain_dt_ms * 1e-3 / sim_timestep)  # 200 steps
    viewer_sync_interval = 160  # ~60 fps

    # ---------------------------------------------------------------
    print("Loading connectome...")
    connectome = Connectome(brain_data_dir)

    sensory_ids = [s for s in SUGAR_SENSING_RIGHT_IDS
                   if s in connectome._flyid_to_idx]
    walking_ids = [MN9_FLYWIRE_ID] + _collect_ids(connectome, WALKING_SEZ_TYPES)
    grooming_ids = _collect_ids(connectome, GROOMING_SEZ_TYPES)
    feeding_ids = _collect_ids(connectome, FEEDING_SEZ_TYPES)

    all_motor_ids: list[int] = []
    seen: set[int] = set()
    for mid in walking_ids + grooming_ids + feeding_ids:
        if mid not in seen and mid in connectome._flyid_to_idx:
            seen.add(mid)
            all_motor_ids.append(mid)

    print(f"  {len(sensory_ids)} sensory, {len(all_motor_ids)} motor neurons")

    # ---------------------------------------------------------------
    print("Building brain model...")
    brain = OnlineBrainModel(
        connectome,
        sensory_neuron_ids=sensory_ids,
        motor_neuron_ids=all_motor_ids,
        dt_ms=0.1,
        use_subnetwork=True,
        n_hops=2,
        min_hop2_weight=3.0,
        max_subnetwork_neurons=35_000,
        spike_window_ms=50.0,
    )
    brain.setup()
    print(f"  Subnetwork: {brain.n_neurons:,} neurons")

    # ---------------------------------------------------------------
    print("Building fly model...")
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
    fly.add_tracking_camera()

    world = NatureWorld()
    world.add_fly(fly, [0, 0, 0.7],
                  Rotation3D(format="quat", values=[1, 0, 0, 0]))

    sim = Simulation(world)
    fly_name = fly.name

    # ---------------------------------------------------------------
    # Bridge with per-leg motor control
    bridge = SensorimotorBridge(
        brain, connectome,
        sensory_gain=5.0, motor_speed_gain=0.002,
        motor_turn_gain=0.005, base_excitation_mv=35.0,
    )

    # Brain-driven behaviour controller (replaces fixed schedule!)
    beh_ctrl = BehaviorController(
        brain, connectome,
        walking_ids=walking_ids,
        grooming_ids=grooming_ids,
        feeding_ids=feeding_ids,
        hold_time_s=1.5,
        grooming_threshold=0.2,   # low — grooming pool has few neurons
        feeding_threshold=0.15,
    )

    # Sensory event generator — simulates environment
    events = SensoryEventGenerator(
        n_sensory=len(sensory_ids),
        event_interval_s=3.0,     # events every ~3s (was 5s)
        event_duration_s=2.5,
        irritation_strength_mv=30.0,
        sugar_strength_mv=25.0,
        seed=42,
    )
    # Force first event to happen within 1 second
    events._next_event_at = 1.0

    # Brain activity visualizer (separate window)
    brain_viz = BrainVisualizer(brain, beh_ctrl, update_every=2)

    walk_cpg = WalkingPatternGenerator()
    groom_prog = GroomingProgram(grooming_freq=3.0)
    feed_prog = FeedingProgram(tap_freq=2.0)

    # ---------------------------------------------------------------
    sim.reset()
    brain.reset()
    walk_cpg.reset()
    groom_prog.reset()
    feed_prog.reset()
    beh_ctrl.reset()
    events.reset(seed=42)
    sim.warmup(duration_s=0.02)
    sim.set_leg_adhesion_states(fly_name, np.ones(6, dtype=np.bool_))

    leg_speeds = np.full(6, 0.5)
    joint_offsets = np.zeros((6, 7))
    turn = 0.0
    current_behavior = BehaviorState.WALKING

    print()
    print("=" * 58)
    print("  Brain-driven fly — close the viewer window to stop")
    print()
    print("  The brain now controls:")
    print("    • Behaviour switching (walk/groom/feed)")
    print("    • Per-leg stepping speed (6 independent legs)")
    print("    • Joint posture modulation (stride/height)")
    print()
    print("  Environmental events trigger behaviour changes:")
    print("    • Antenna dust → grooming")
    print("    • Sugar encounter → feeding")
    print("    • Quiet periods → walking")
    print("=" * 58)

    with mujoco.viewer.launch_passive(sim.mj_model, sim.mj_data) as viewer:
        step_idx = 0
        last_behavior = None
        last_event = None
        import time as _time
        _brain_times = []

        while viewer.is_running():
            for _ in range(viewer_sync_interval):
                t = step_idx * sim_timestep

                # --- Brain update ---
                if step_idx % brain_update_interval == 0:
                    _t0 = _time.perf_counter()
                    # 1) Read body sensors
                    joint_angles = sim.get_joint_angles(fly_name)
                    contact_info = sim.get_ground_contact_info(fly_name)
                    contact_active = contact_info[0]
                    forces = contact_info[1]
                    body_positions = sim.get_body_positions(fly_name)

                    # 2) Compute baseline sensory currents
                    currents = bridge.sensors_to_currents(
                        contact_active, forces, joint_angles, body_positions
                    )

                    # 3) Add environmental event stimulation to sensory neurons
                    event_extra = events.step(brain_dt_ms * 1e-3)
                    currents += event_extra

                    # 3b) Also inject current directly into the relevant
                    #     motor neuron pool — simulates descending commands
                    #     from higher brain regions not in our subnetwork
                    ev = events.current_event
                    if ev == "antenna_irritation":
                        brain.inject_motor_current(
                            beh_ctrl._grooming_local, 20.0)
                    elif ev == "sugar_detection":
                        brain.inject_motor_current(
                            beh_ctrl._feeding_local, 18.0)

                    # 4) Inject into brain and step
                    brain.inject_sensory_current(currents)
                    brain.step(brain_dt_ms)

                    _brain_times.append(_time.perf_counter() - _t0)
                    if len(_brain_times) % 10 == 0:
                        avg = sum(_brain_times[-10:]) / 10
                        print(f"  [PERF] brain step avg: {avg*1000:.0f} ms  "
                              f"(sim {t:.2f}s after {len(_brain_times)} updates)")

                    # 5) Brain decides behaviour
                    current_behavior = beh_ctrl.update(brain_dt_ms * 1e-3, event=ev)

                    # 6) Per-leg motor control
                    motor_rates = brain.get_motor_spike_rates()
                    leg_speeds, joint_offsets, turn = \
                        bridge.motor_rates_to_leg_commands(motor_rates)

                    # Log behaviour and event changes
                    ev = events.current_event
                    if ev != last_event:
                        event_label = {"baseline": "quiet",
                                       "antenna_irritation": "ANTENNA DUST",
                                       "sugar_detection": "SUGAR FOUND"}
                        print(f"  [{t:6.1f}s] env: {event_label.get(ev, ev)}")
                        last_event = ev

                    if current_behavior != last_behavior:
                        print(f"  [{t:6.1f}s] brain → {current_behavior.value}")
                        last_behavior = current_behavior

                    # Update brain visualizer
                    brain_viz.update(ev, current_behavior)

                # --- Motor programme ---
                if current_behavior == BehaviorState.WALKING:
                    angles = walk_cpg.step_per_leg(
                        sim_timestep, leg_speeds=leg_speeds,
                        turn=turn, joint_offsets=joint_offsets)
                    adhesion = walk_cpg.get_adhesion_states().astype(float)
                elif current_behavior == BehaviorState.GROOMING:
                    angles = groom_prog.step(sim_timestep)
                    adhesion = groom_prog.get_adhesion_states().astype(float)
                elif current_behavior == BehaviorState.FEEDING:
                    angles = feed_prog.step(sim_timestep, speed=0.15, turn=turn)
                    adhesion = feed_prog.get_adhesion_states().astype(float)
                else:
                    angles = walk_cpg.step_per_leg(
                        sim_timestep, leg_speeds=leg_speeds,
                        turn=turn, joint_offsets=joint_offsets)
                    adhesion = walk_cpg.get_adhesion_states().astype(float)

                target = reorder_cpg_to_actuator(angles, fly, ActuatorType.POSITION)
                sim.set_actuator_inputs(fly_name, ActuatorType.POSITION, target)
                sim.set_leg_adhesion_states(fly_name, adhesion)

                sim.step()
                step_idx += 1

            viewer.sync()

    brain_viz.close()


if __name__ == "__main__":
    main()
