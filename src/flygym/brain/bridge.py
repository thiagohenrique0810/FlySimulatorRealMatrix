"""Sensorimotor bridge between the brain model and FlyGym body simulation.

Converts FlyGym sensor readings into neural currents for the brain model,
and converts motor neuron firing rates into joint-angle targets for the body.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from flygym.brain.connectome import Connectome
from flygym.brain.lif_model import OnlineBrainModel


# ---------------------------------------------------------------------------
# Walking-pattern generator (CPG-like, parametric)
# ---------------------------------------------------------------------------


class WalkingPatternGenerator:
    """A simple CPG-like generator that produces tripod-gait joint angles.

    The pattern is parameterized by walking speed (frequency) and turning
    (left/right amplitude modulation).  Joint angle trajectories are based
    on sinusoidal approximations of the experimental walking data with
    biologically plausible ranges for *Drosophila melanogaster*.

    The six legs follow a **tripod gait** with two anti-phase groups:
    - Group A (phase = 0):  lf, rm, lh
    - Group B (phase = π):  rf, lm, rh

    For each leg, seven DOFs are produced:
    ``(coxa_pitch, coxa_roll, coxa_yaw, femur_pitch, femur_roll,
    tibia_pitch, tarsus1_pitch)``

    Args:
        base_frequency: Stepping frequency in Hz at ``speed=1.0``.
        n_dofs_per_leg: Number of degrees of freedom per leg (default 7).
    """

    # Leg names in the canonical order used by FlyGym v2
    LEG_NAMES = ["lf", "lm", "lh", "rf", "rm", "rh"]

    # Tripod gait phase offsets (radians)
    TRIPOD_PHASES = {
        "lf": 0.0,
        "rm": 0.0,
        "lh": 0.0,
        "rf": np.pi,
        "lm": np.pi,
        "rh": np.pi,
    }

    # Neutral (resting) joint angles per DOF (radians), shared for all legs.
    # Derived from experimental walking data (MotionSnippet mean values).
    # Order: coxa_yaw, coxa_pitch, coxa_roll, femur_pitch, femur_roll,
    #        tibia_pitch, tarsus1_pitch
    NEUTRAL_ANGLES = np.array([0.175, 1.663, -0.041, -2.051, 0.114, 1.643, -0.405])

    # Oscillation amplitudes per DOF (radians) during walking.
    # Derived from experimental walking data (half peak-to-peak ranges).
    WALK_AMPLITUDES = np.array([0.29, 0.50, 0.31, 0.67, 0.65, 0.93, 0.62])

    def __init__(
        self,
        base_frequency: float = 12.0,
        n_dofs_per_leg: int = 7,
    ) -> None:
        self.base_frequency = base_frequency
        self.n_dofs_per_leg = n_dofs_per_leg

        # Internal phase accumulators (one per leg)
        self._phases: dict[str, float] = {
            leg: self.TRIPOD_PHASES[leg] for leg in self.LEG_NAMES
        }

    def step(
        self,
        dt: float,
        speed: float = 1.0,
        turn: float = 0.0,
    ) -> np.ndarray:
        """Advance the CPG by *dt* seconds and return joint angle targets.

        Args:
            dt: Time step in seconds.
            speed: Walking speed multiplier (0 = standing, 1 = normal, >1 = fast).
            turn: Turning signal in [-1, 1].
                  Positive = turn right (left legs faster),
                  Negative = turn left (right legs faster).

        Returns:
            Joint angle targets of shape ``(6, 7)`` in radians, with legs
            ordered as ``["lf", "lm", "lh", "rf", "rm", "rh"]``.
        """
        angles = np.zeros((6, self.n_dofs_per_leg))

        for i, leg in enumerate(self.LEG_NAMES):
            # Determine frequency modulation for turning
            side = leg[0]  # 'l' or 'r'
            if side == "l":
                freq_mod = 1.0 + turn * 0.3
                amp_mod = 1.0 + turn * 0.2
            else:
                freq_mod = 1.0 - turn * 0.3
                amp_mod = 1.0 - turn * 0.2

            freq = self.base_frequency * speed * freq_mod
            amp_scale = speed * max(amp_mod, 0.1)

            # Advance phase
            self._phases[leg] += 2.0 * np.pi * freq * dt
            phase = self._phases[leg]

            # Generate joint angles: neutral + amplitude * sin(phase + dof_offset)
            for d in range(self.n_dofs_per_leg):
                dof_phase_offset = d * 0.15  # slight phase offset between DOFs
                oscillation = (
                    self.WALK_AMPLITUDES[d]
                    * amp_scale
                    * np.sin(phase + dof_phase_offset)
                )
                angles[i, d] = self.NEUTRAL_ANGLES[d] + oscillation

        return angles

    def get_adhesion_states(self) -> np.ndarray:
        """Compute leg adhesion states based on current gait phase.

        Returns:
            Boolean array of shape ``(6,)``:  True = stance (adhesion on),
            False = swing (adhesion off).
        """
        states = np.ones(6, dtype=np.bool_)
        for i, leg in enumerate(self.LEG_NAMES):
            # Stance when sin(phase) >= 0 (half of the cycle)
            phase_mod = self._phases[leg] % (2.0 * np.pi)
            states[i] = phase_mod < np.pi
        return states

    def reset(self) -> None:
        """Reset all phases to initial tripod configuration."""
        self._phases = {
            leg: self.TRIPOD_PHASES[leg] for leg in self.LEG_NAMES
        }


# ---------------------------------------------------------------------------
# Sensorimotor bridge
# ---------------------------------------------------------------------------


class SensorimotorBridge:
    """Bidirectional mapping between FlyGym sensors and brain neurons.

    Converts FlyGym sensor outputs (contacts, joint angles, body pose) into
    depolarising currents for sensory neurons, and converts motor/descending
    neuron firing rates into walking speed and turning commands that drive
    a :class:`WalkingPatternGenerator`.

    Args:
        brain: An initialised :class:`OnlineBrainModel`.
        connectome: The :class:`Connectome` backing the brain.
        sensory_gain: Scaling factor from sensor values to neural current (mV).
        motor_speed_gain: Scaling factor from motor neuron Hz to walking speed.
        motor_turn_gain: Scaling factor from left/right rate asymmetry to turn.
        base_excitation_mv: Tonic excitation applied to sensory neurons (mV).
            Must exceed the LIF threshold gap (v_th - v_0 = 7 mV by default)
            to generate spiking.  Typical values: 10–15 mV.
    """

    def __init__(
        self,
        brain: OnlineBrainModel,
        connectome: Connectome,
        *,
        sensory_gain: float = 2.0,
        motor_speed_gain: float = 0.005,
        motor_turn_gain: float = 0.01,
        base_excitation_mv: float = 12.0,
    ) -> None:
        self.brain = brain
        self.connectome = connectome
        self.sensory_gain = sensory_gain
        self.motor_speed_gain = motor_speed_gain
        self.motor_turn_gain = motor_turn_gain
        self.base_excitation_mv = base_excitation_mv

        self._n_sensory = brain.n_sensory
        self._n_motor = brain.n_motor

    # ------------------------------------------------------------------
    # Sensory → Neural
    # ------------------------------------------------------------------

    def sensors_to_currents(
        self,
        contact_active: np.ndarray,
        forces: np.ndarray,
        joint_angles: np.ndarray,
        body_positions: np.ndarray,
    ) -> np.ndarray:
        """Convert FlyGym sensor data into currents for sensory neurons.

        The sensory neurons are evenly divided into three groups:
        1. **Mechanosensory** — driven by ground contact forces
        2. **Proprioceptive** — driven by joint angle magnitudes
        3. **Vestibular** — driven by body tilt / orientation

        Each group receives a current proportional to a summary statistic
        of the relevant sensor reading plus the tonic excitation.

        Args:
            contact_active: Shape ``(6,)`` — ground contact flags per leg.
            forces: Shape ``(6, 3)`` — contact forces per leg.
            joint_angles: Shape ``(n_dofs,)`` — joint angles in radians.
            body_positions: Shape ``(n_bodies, 3)`` — body positions.

        Returns:
            Array of shape ``(n_sensory,)`` with currents in mV.
        """
        currents = np.full(self._n_sensory, self.base_excitation_mv)

        if self._n_sensory == 0:
            return currents

        # Split sensory neurons into 3 groups
        group_size = max(self._n_sensory // 3, 1)
        mechano_end = group_size
        proprio_end = group_size * 2

        # 1. Mechanosensory: total contact force magnitude
        total_force = np.sqrt(np.sum(forces ** 2, axis=1))  # (6,)
        mechano_signal = np.mean(total_force) * self.sensory_gain * 0.001
        currents[:mechano_end] += mechano_signal

        # 2. Proprioceptive: mean absolute joint deviation from neutral
        if len(joint_angles) > 0:
            proprio_signal = np.mean(np.abs(joint_angles)) * self.sensory_gain
            currents[mechano_end:proprio_end] += proprio_signal

        # 3. Vestibular: body height and tilt
        if body_positions is not None and len(body_positions) > 0:
            # Use mean z-position as a proxy for body height
            mean_z = np.mean(body_positions[:, 2])
            vestibular_signal = abs(mean_z - 0.5) * self.sensory_gain
            currents[proprio_end:] += vestibular_signal

        return currents

    # ------------------------------------------------------------------
    # Neural → Motor
    # ------------------------------------------------------------------

    def motor_rates_to_commands(
        self,
        motor_rates: np.ndarray,
    ) -> tuple[float, float]:
        """Convert motor neuron firing rates to walking speed and turning.

        The first half of the motor neurons is assigned to the *left* side,
        the second half to the *right*.  Walking speed is proportional to
        the mean rate; turning is proportional to L–R asymmetry.

        Args:
            motor_rates: Shape ``(n_motor,)`` — firing rates in Hz.

        Returns:
            ``(speed, turn)`` — speed ∈ [0, ∞), turn ∈ [-1, 1].
        """
        if len(motor_rates) == 0:
            return (0.5, 0.0)

        mean_rate = np.mean(motor_rates)
        # Brain-modulated speed on top of a baseline walking speed
        brain_speed = mean_rate * self.motor_speed_gain
        speed = np.clip(0.4 + brain_speed, 0.1, 2.0)

        # Left/right split
        half = max(len(motor_rates) // 2, 1)
        left_rate = np.mean(motor_rates[:half])
        right_rate = np.mean(motor_rates[half:]) if half < len(motor_rates) else left_rate
        diff = (left_rate - right_rate) * self.motor_turn_gain
        turn = np.clip(diff, -1.0, 1.0)

        return (speed, turn)

    # ------------------------------------------------------------------
    # One-step convenience
    # ------------------------------------------------------------------

    def update_brain_and_get_commands(
        self,
        contact_active: np.ndarray,
        forces: np.ndarray,
        joint_angles: np.ndarray,
        body_positions: np.ndarray,
        brain_dt_ms: float,
    ) -> tuple[float, float]:
        """Run one full sense → brain → act cycle.

        Returns:
            ``(speed, turn)`` for the walking pattern generator.
        """
        # Sense → Neural
        currents = self.sensors_to_currents(
            contact_active, forces, joint_angles, body_positions
        )
        self.brain.inject_sensory_current(currents)

        # Brain step
        self.brain.step(brain_dt_ms)

        # Neural → Motor
        motor_rates = self.brain.get_motor_spike_rates()
        return self.motor_rates_to_commands(motor_rates)


# ---------------------------------------------------------------------------
# Utility: reorder CPG output to match fly actuator DOF order
# ---------------------------------------------------------------------------


def reorder_cpg_to_actuator(
    cpg_angles: np.ndarray,
    fly,
    actuator_type,
) -> np.ndarray:
    """Reorder ``(6, 7)`` CPG output to the flat actuator-order array.

    The CPG produces angles in the order ``["lf", "lm", "lh", "rf", "rm", "rh"]``
    with 7 DOFs per leg ``(coxa_pitch, coxa_roll, coxa_yaw, femur_pitch,
    femur_roll, tibia_pitch, tarsus1_pitch)``.

    This function maps them to the order expected by
    ``fly.get_actuated_jointdofs_order(actuator_type)``.

    Args:
        cpg_angles: Shape ``(6, 7)`` from :class:`WalkingPatternGenerator`.
        fly: A :class:`~flygym.compose.Fly` instance with actuators.
        actuator_type: The actuator type.

    Returns:
        Flat array of shape ``(n_actuated_dofs,)`` matching the DOF order
        expected by ``sim.set_actuator_inputs()``.
    """
    from flygym.compose.fly import ActuatorType

    actuator_type = ActuatorType(actuator_type)
    dof_order = fly.get_actuated_jointdofs_order(actuator_type)

    cpg_leg_order = WalkingPatternGenerator.LEG_NAMES  # lf, lm, lh, rf, rm, rh
    cpg_dof_axes = ["yaw", "pitch", "roll", "pitch", "roll", "pitch", "pitch"]
    cpg_dof_links = [
        ("thorax", "coxa"),
        ("thorax", "coxa"),
        ("thorax", "coxa"),
        ("coxa", "trochanterfemur"),
        ("coxa", "trochanterfemur"),
        ("trochanterfemur", "tibia"),
        ("tibia", "tarsus1"),
    ]

    # Build lookup: (leg_pos, parent_link, child_link, axis) → cpg index
    cpg_lookup: dict[tuple[str, str, str, str], tuple[int, int]] = {}
    for leg_i, leg in enumerate(cpg_leg_order):
        for dof_j in range(7):
            parent, child = cpg_dof_links[dof_j]
            axis = cpg_dof_axes[dof_j]
            key = (leg, parent, child, axis)
            cpg_lookup[key] = (leg_i, dof_j)

    result = np.zeros(len(dof_order))
    for out_idx, dof in enumerate(dof_order):
        leg_pos = dof.child.pos  # e.g. "lf"
        parent_link = dof.parent.link  # e.g. "thorax"
        child_link = dof.child.link  # e.g. "coxa"
        axis_name = dof.axis.value  # e.g. "pitch"
        key = (leg_pos, parent_link, child_link, axis_name)
        if key in cpg_lookup:
            li, dj = cpg_lookup[key]
            result[out_idx] = cpg_angles[li, dj]
        # else: leave at 0 (non-leg DOFs, if any)

    return result
