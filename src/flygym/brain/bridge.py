"""Sensorimotor bridge between the brain model and FlyGym body simulation.

Converts FlyGym sensor readings into neural currents for the brain model,
and converts motor neuron firing rates into joint-angle targets for the body.
Supports three behaviours: walking, antenna grooming, and food seeking.
"""

from __future__ import annotations

from enum import Enum
from typing import Sequence

import numpy as np

from flygym.brain.connectome import Connectome
from flygym.brain.lif_model import OnlineBrainModel


# ---------------------------------------------------------------------------
# Walking-pattern generator — data-driven (replays experimental recording)
# ---------------------------------------------------------------------------


class WalkingPatternGenerator:
    """Data-driven walking pattern generator.

    Instead of synthesising joint angles from sinusoids, this generator
    replays the experimentally recorded *Drosophila* walking snippet
    shipped with ``flygym_demo``.  The phase pointer advances at a rate
    proportional to the ``speed`` parameter, and turning is achieved by
    modulating the playback speed of left vs right legs.

    This guarantees that the inter-DOF coordination, amplitude ranges and
    stance/swing timing are always biologically realistic.

    Args:
        base_frequency: Nominal stepping frequency (Hz) at speed = 1.0.
            The experimental data contains ~12 Hz stepping.
    """

    LEG_NAMES = ["lf", "lm", "lh", "rf", "rm", "rh"]

    # Tripod groups for adhesion: group A swings while group B is in stance
    _TRIPOD_A = {0, 4, 2}   # lf, rm, lh  (indices)
    _TRIPOD_B = {3, 1, 5}   # rf, lm, rh

    def __init__(self, base_frequency: float = 12.0) -> None:
        self.base_frequency = base_frequency

        # Load the experimental walking data
        from flygym_demo.spotlight_data import MotionSnippet
        snippet = MotionSnippet()
        # shape (n_frames, 6, 7), legs order matches LEG_NAMES,
        # DOFs per leg: (coxa_pitch, coxa_roll, coxa_yaw,
        #                trochanterfemur_pitch, trochanterfemur_roll,
        #                tibia_pitch, tarsus1_pitch)
        self._ref_angles: np.ndarray = snippet.joint_angles.copy()
        self._ref_fps: float = float(snippet.data_fps)
        self._n_frames: int = self._ref_angles.shape[0]

        # Per-leg, per-DOF min/max from experimental data (for clamping)
        self._ref_min: np.ndarray = np.min(self._ref_angles, axis=0)  # (6, 7)
        self._ref_max: np.ndarray = np.max(self._ref_angles, axis=0)  # (6, 7)
        # Add a small margin so brain offsets have some room
        margin = 0.05  # ~3 degrees
        self._ref_min -= margin
        self._ref_max += margin

        # Phase pointer (fractional frame index, wraps around)
        self._phase: float = 0.0

        # Per-leg phase offsets for turning (left legs vs right legs)
        self._leg_phases: np.ndarray = np.zeros(6)

    # DOF ordering in the reference data (matches MotionSnippet.dofs_per_leg)
    DOF_SPEC = [
        ("thorax", "coxa", "pitch"),
        ("thorax", "coxa", "roll"),
        ("thorax", "coxa", "yaw"),
        ("coxa", "trochanterfemur", "pitch"),
        ("coxa", "trochanterfemur", "roll"),
        ("trochanterfemur", "tibia", "pitch"),
        ("tibia", "tarsus1", "pitch"),
    ]

    def step(
        self,
        dt: float,
        speed: float = 1.0,
        turn: float = 0.0,
    ) -> np.ndarray:
        """Advance the pattern by *dt* seconds and return joint angles.

        Args:
            dt: Physics timestep in seconds.
            speed: Walking speed multiplier (0 = stationary, 1 = normal).
            turn: Turning signal in [-1, 1]. Positive = turn right.

        Returns:
            Joint angles of shape ``(6, 7)`` in the same DOF order as the
            experimental data.
        """
        leg_speeds = np.full(6, speed)
        return self.step_per_leg(dt, leg_speeds=leg_speeds, turn=turn)

    def step_per_leg(
        self,
        dt: float,
        leg_speeds: np.ndarray,
        turn: float = 0.0,
        joint_offsets: np.ndarray | None = None,
    ) -> np.ndarray:
        """Advance with per-leg speed control and optional joint offsets.

        This is the brain-enhanced version: instead of one global speed,
        each leg has its own stepping speed driven by its motor neuron pool.

        Args:
            dt: Physics timestep in seconds.
            leg_speeds: Shape ``(6,)`` — speed multiplier for each leg.
            turn: Turning signal in [-1, 1]. Positive = turn right.
            joint_offsets: Optional shape ``(6, 7)`` — angle offsets added
                to the replayed experimental data (brain-modulated posture).

        Returns:
            Joint angles of shape ``(6, 7)``.
        """
        frames_per_sec = self._ref_fps  # 330 fps in the recording

        # Advance per-leg phases with individual speeds and turning
        for i, leg in enumerate(self.LEG_NAMES):
            side = leg[0]
            if side == "l":
                turn_mod = 1.0 + turn * 0.3
            else:
                turn_mod = 1.0 - turn * 0.3
            self._leg_phases[i] += frames_per_sec * leg_speeds[i] * turn_mod * dt

        # Also advance the global phase (used for adhesion)
        self._phase += frames_per_sec * float(np.mean(leg_speeds)) * dt

        # Sample angles for each leg from the reference data via interpolation
        angles = np.zeros((6, 7))
        for i in range(6):
            ph = self._leg_phases[i] % self._n_frames
            f0 = int(ph) % self._n_frames
            f1 = (f0 + 1) % self._n_frames
            frac = ph - int(ph)
            angles[i] = (1.0 - frac) * self._ref_angles[f0, i] + frac * self._ref_angles[f1, i]

        # Apply brain-driven joint offsets
        if joint_offsets is not None:
            angles += joint_offsets

        # Clamp to the range seen in the experimental data so legs
        # never go into unnatural poses
        angles = np.clip(angles, self._ref_min, self._ref_max)

        return angles

    def get_adhesion_states(self) -> np.ndarray:
        """Return per-leg adhesion states based on the reference data.

        Uses the tibia pitch (DOF 5) to detect stance vs swing:
        when tibia is extended (high pitch = leg down) → stance.
        """
        states = np.ones(6, dtype=np.bool_)
        for i in range(6):
            ph = self._leg_phases[i] % self._n_frames
            f0 = int(ph) % self._n_frames
            f1 = (f0 + 1) % self._n_frames
            frac = ph - int(ph)
            tibia_pitch = ((1.0 - frac) * self._ref_angles[f0, i, 5]
                           + frac * self._ref_angles[f1, i, 5])
            # Stance when tibia is more extended (above median)
            median_tp = np.median(self._ref_angles[:, i, 5])
            states[i] = tibia_pitch >= median_tp
        return states

    def reset(self) -> None:
        """Reset phase to the beginning of the recording."""
        self._phase = 0.0
        self._leg_phases = np.zeros(6)


# ---------------------------------------------------------------------------
# Behaviour state machine
# ---------------------------------------------------------------------------


class BehaviorState(Enum):
    """Active behaviour of the fly."""
    WALKING = "walking"
    GROOMING = "grooming"
    FEEDING = "feeding"
    SEARCHING = "searching"
    ESCAPE = "escape"
    FREEZING = "freezing"
    BACKWARD = "backward"


# ---------------------------------------------------------------------------
# Antenna-grooming motor programme
# ---------------------------------------------------------------------------


class GroomingProgram:
    """Generates front-leg grooming trajectories.

    During grooming the front legs (LF, RF) lift toward the head and
    oscillate in a scraping motion while the four remaining legs hold a
    stable stance position derived from the experimental walking data.

    Args:
        grooming_freq: Scraping frequency in Hz (typically ~3 Hz).
    """

    LEG_NAMES = WalkingPatternGenerator.LEG_NAMES
    DOF_SPEC = WalkingPatternGenerator.DOF_SPEC

    def __init__(self, grooming_freq: float = 3.0) -> None:
        self.grooming_freq = grooming_freq
        self._phase: float = 0.0

        # Load walking data to get stance angles for mid/hind legs
        from flygym_demo.spotlight_data import MotionSnippet
        snippet = MotionSnippet()
        ref = snippet.joint_angles  # (660, 6, 7)

        # Stance = mean angles (stable standing approximation)
        self._stance_angles: np.ndarray = np.mean(ref, axis=0)  # (6, 7)

        # ----- Front-leg grooming key-poses (lf) -----
        # Two poses that the front legs oscillate between:
        # "reach" = legs stretched forward & up toward antenna
        # "scrape" = legs pulled slightly back & down
        lf_mean = self._stance_angles[0]  # reference

        # Reach pose: coxa forward, femur extended, tibia curled up
        self._reach_lf = lf_mean.copy()
        self._reach_lf[0] = 0.75   # coxa_pitch: swing forward
        self._reach_lf[1] = 0.45   # coxa_roll: bring inward
        self._reach_lf[3] = -1.3   # femur_pitch: extend up
        self._reach_lf[5] = 0.7    # tibia_pitch: flex toward head
        self._reach_lf[6] = -0.9   # tarsus: curl to touch antenna

        # Scrape pose: slight retraction (rubbing motion)
        self._scrape_lf = lf_mean.copy()
        self._scrape_lf[0] = 0.55  # coxa_pitch: pull back a bit
        self._scrape_lf[1] = 0.50  # coxa_roll: still inward
        self._scrape_lf[3] = -1.5  # femur_pitch: slightly lower
        self._scrape_lf[5] = 1.0   # tibia_pitch: slightly extended
        self._scrape_lf[6] = -0.7  # tarsus: partial release

        # Mirror for rf (roll and yaw signs flip)
        self._reach_rf = self._reach_lf.copy()
        self._reach_rf[1] = self._stance_angles[3, 1] - (self._stance_angles[0, 1] - self._reach_lf[1])
        self._reach_rf[2] = -self._reach_lf[2] if abs(self._reach_lf[2]) > 0.01 else self._reach_lf[2]

        self._scrape_rf = self._scrape_lf.copy()
        self._scrape_rf[1] = self._stance_angles[3, 1] - (self._stance_angles[0, 1] - self._scrape_lf[1])
        self._scrape_rf[2] = -self._scrape_lf[2] if abs(self._scrape_lf[2]) > 0.01 else self._scrape_lf[2]

    def step(self, dt: float) -> np.ndarray:
        """Advance grooming motion by *dt* seconds.

        Returns:
            Joint angles ``(6, 7)`` — same shape as
            :class:`WalkingPatternGenerator`.
        """
        self._phase += dt * self.grooming_freq * 2.0 * np.pi

        # Smooth oscillation factor 0→1→0
        t = 0.5 * (1.0 + np.sin(self._phase))

        angles = self._stance_angles.copy()
        # Front legs: interpolate between reach and scrape
        angles[0] = (1.0 - t) * self._scrape_lf + t * self._reach_lf
        angles[3] = (1.0 - t) * self._scrape_rf + t * self._reach_rf

        return angles

    def get_adhesion_states(self) -> np.ndarray:
        """During grooming, front legs are OFF ground, rest are ON."""
        return np.array([False, True, True, False, True, True])

    def reset(self) -> None:
        self._phase = 0.0


# ---------------------------------------------------------------------------
# Food-seeking / feeding motor programme
# ---------------------------------------------------------------------------


class FeedingProgram:
    """Generates food-seeking and proboscis extension behaviour.

    The fly slows down and makes exploratory turns.  Periodically the
    front legs lift slightly (chemosensory tapping) and the walking
    pattern alternates between slow forward motion and pausing.

    Args:
        tap_freq: Front-leg tapping frequency (Hz).
    """

    LEG_NAMES = WalkingPatternGenerator.LEG_NAMES
    DOF_SPEC = WalkingPatternGenerator.DOF_SPEC

    def __init__(self, tap_freq: float = 2.0) -> None:
        self.tap_freq = tap_freq
        self._cpg = WalkingPatternGenerator()
        self._tap_phase: float = 0.0

    def step(
        self, dt: float, speed: float = 0.15, turn: float = 0.0
    ) -> np.ndarray:
        """Advance feeding motion.

        Returns:
            Joint angles ``(6, 7)``.
        """
        self._tap_phase += dt * self.tap_freq * 2.0 * np.pi
        tap = 0.5 * (1.0 + np.sin(self._tap_phase))

        # Slow walking with exploration turns
        explore_turn = turn + 0.3 * np.sin(self._tap_phase * 0.37)
        angles = self._cpg.step(dt, speed=speed, turn=np.clip(explore_turn, -1, 1))

        # Front-leg tapping: periodically lift front legs slightly
        if tap > 0.7:
            lift = (tap - 0.7) / 0.3  # 0→1 during top 30%
            for i in [0, 3]:  # lf, rf
                angles[i, 0] += 0.2 * lift   # coxa pitch forward
                angles[i, 3] += 0.3 * lift   # femur pitch up
                angles[i, 5] -= 0.3 * lift   # tibia flex
                angles[i, 6] -= 0.2 * lift   # tarsus tap

        return angles

    def get_adhesion_states(self) -> np.ndarray:
        """During feeding, use the walking CPG's adhesion with front legs
        sometimes OFF during tapping."""
        states = self._cpg.get_adhesion_states()
        tap = 0.5 * (1.0 + np.sin(self._tap_phase))
        if tap > 0.7:
            states[0] = False  # lf
            states[3] = False  # rf
        return states

    def reset(self) -> None:
        self._cpg.reset()
        self._tap_phase = 0.0


# ---------------------------------------------------------------------------
# Odor search programme — chemotaxis-inspired zigzag walk
# ---------------------------------------------------------------------------


class OdorSearchProgram:
    """Generates odor-tracking search behaviour.

    When the fly detects an odor, it performs a chemotaxis-inspired
    pattern: moderate-speed walking with alternating left/right turns
    (casting) to locate the odor source.  The antenna oscillate slightly
    as if sampling the air.

    Args:
        cast_freq: Frequency of left-right casting turns (Hz).
        cast_amplitude: Maximum turn magnitude during casting.
    """

    LEG_NAMES = WalkingPatternGenerator.LEG_NAMES
    DOF_SPEC = WalkingPatternGenerator.DOF_SPEC

    def __init__(
        self, cast_freq: float = 1.5, cast_amplitude: float = 0.6
    ) -> None:
        self.cast_freq = cast_freq
        self.cast_amplitude = cast_amplitude
        self._cpg = WalkingPatternGenerator()
        self._cast_phase: float = 0.0

    def step(
        self, dt: float, speed: float = 0.6, turn: float = 0.0
    ) -> np.ndarray:
        """Advance odor-search motion.

        Returns:
            Joint angles ``(6, 7)``.
        """
        self._cast_phase += dt * self.cast_freq * 2.0 * np.pi

        # Casting turn: sinusoidal left-right sweeps
        cast_turn = self.cast_amplitude * np.sin(self._cast_phase)
        total_turn = np.clip(turn + cast_turn, -1.0, 1.0)

        angles = self._cpg.step(dt, speed=speed, turn=total_turn)

        # Subtle antenna sampling: front legs lift slightly during casting peaks
        cast_mag = abs(np.sin(self._cast_phase))
        if cast_mag > 0.8:
            lift = (cast_mag - 0.8) / 0.2  # 0→1
            for i in [0, 3]:  # lf, rf
                angles[i, 0] += 0.10 * lift   # coxa pitch forward (antenna reach)
                angles[i, 3] += 0.15 * lift   # femur pitch up slightly

        return angles

    def get_adhesion_states(self) -> np.ndarray:
        """During searching, use normal walking adhesion."""
        return self._cpg.get_adhesion_states()

    def reset(self) -> None:
        self._cpg.reset()
        self._cast_phase = 0.0


# ---------------------------------------------------------------------------
# Escape / startle programme — fast burst run + random turn
# ---------------------------------------------------------------------------


class EscapeProgram:
    """Generates a fast escape response.

    The fly accelerates to maximum speed and executes a sharp random turn
    away from the perceived threat.  The burst decays over ~1 s to a
    moderately fast run before the controller switches back to walking.

    Args:
        burst_speed: Peak speed multiplier during the initial startle.
        turn_magnitude: How sharply the fly turns at escape onset.
    """

    LEG_NAMES = WalkingPatternGenerator.LEG_NAMES
    DOF_SPEC = WalkingPatternGenerator.DOF_SPEC

    def __init__(
        self, burst_speed: float = 2.0, turn_magnitude: float = 0.8
    ) -> None:
        self.burst_speed = burst_speed
        self.turn_magnitude = turn_magnitude
        self._cpg = WalkingPatternGenerator()
        self._elapsed: float = 0.0
        self._escape_turn: float = 0.0
        self._rng = np.random.default_rng()

    def start(self) -> None:
        """Call once when escape begins to pick a random dodge direction."""
        self._elapsed = 0.0
        self._escape_turn = self._rng.choice([-1.0, 1.0]) * self.turn_magnitude

    def step(
        self, dt: float, speed: float = 1.0, turn: float = 0.0
    ) -> np.ndarray:
        """Advance escape motion.

        Returns:
            Joint angles ``(6, 7)``.
        """
        self._elapsed += dt

        # Burst decays exponentially: fast start, gradual slowdown
        decay = np.exp(-self._elapsed * 2.0)
        esc_speed = speed + (self.burst_speed - speed) * decay
        esc_turn = np.clip(turn + self._escape_turn * decay, -1.0, 1.0)

        angles = self._cpg.step(dt, speed=esc_speed, turn=esc_turn)

        # Legs spread slightly for stability during fast running
        if decay > 0.3:
            spread = decay * 0.1
            for i in range(6):
                angles[i, 1] += spread * (1 if i < 3 else -1)  # coxa roll outward

        return angles

    def get_adhesion_states(self) -> np.ndarray:
        return self._cpg.get_adhesion_states()

    def reset(self) -> None:
        self._cpg.reset()
        self._elapsed = 0.0
        self._escape_turn = 0.0


# ---------------------------------------------------------------------------
# Freezing programme — complete immobility
# ---------------------------------------------------------------------------


class FreezingProgram:
    """Generates freezing (tonic immobility) posture.

    All legs hold a stable stance position derived from the experimental
    walking data.  No motion occurs.  This is the anti-predator
    "play dead" response.
    """

    LEG_NAMES = WalkingPatternGenerator.LEG_NAMES
    DOF_SPEC = WalkingPatternGenerator.DOF_SPEC

    def __init__(self) -> None:
        from flygym_demo.spotlight_data import MotionSnippet
        snippet = MotionSnippet()
        # Mean of experimental walking = a stable standing posture
        self._stance_angles: np.ndarray = np.mean(
            snippet.joint_angles, axis=0
        )  # (6, 7)

        # Lower the body slightly (crouching freeze)
        for i in range(6):
            self._stance_angles[i, 3] += 0.15   # femur pitch: crouch
            self._stance_angles[i, 5] += 0.10   # tibia pitch: flatten

    def step(self, dt: float, **kwargs) -> np.ndarray:
        """Return the frozen stance posture (unchanged every step)."""
        return self._stance_angles.copy()

    def get_adhesion_states(self) -> np.ndarray:
        """All legs firmly on the ground during freezing."""
        return np.ones(6, dtype=np.bool_)

    def reset(self) -> None:
        pass  # No internal state to reset


# ---------------------------------------------------------------------------
# Backward walking programme — reverse and pivot
# ---------------------------------------------------------------------------


class BackwardProgram:
    """Generates backward walking for obstacle avoidance.

    Plays the walking CPG in reverse (negative speed) with a gradual
    pivot to one side so the fly retreats and reorients.

    Args:
        reverse_speed: Backward walking speed multiplier (positive value).
        pivot_rate: How fast the fly pivots while reversing.
    """

    LEG_NAMES = WalkingPatternGenerator.LEG_NAMES
    DOF_SPEC = WalkingPatternGenerator.DOF_SPEC

    def __init__(
        self, reverse_speed: float = 0.5, pivot_rate: float = 0.4
    ) -> None:
        self.reverse_speed = reverse_speed
        self.pivot_rate = pivot_rate
        self._cpg = WalkingPatternGenerator()
        self._elapsed: float = 0.0
        self._pivot_dir: float = 0.0
        self._rng = np.random.default_rng()

    def start(self) -> None:
        """Call once when backward walking begins to pick pivot direction."""
        self._elapsed = 0.0
        self._pivot_dir = self._rng.choice([-1.0, 1.0])

    def step(
        self, dt: float, speed: float = 0.5, turn: float = 0.0
    ) -> np.ndarray:
        """Advance backward motion.

        Returns:
            Joint angles ``(6, 7)``.
        """
        self._elapsed += dt

        # Walk backward (negative speed through CPG)
        rev_speed = -self.reverse_speed
        # Gradually increase pivot as the fly retreats
        pivot = self._pivot_dir * self.pivot_rate * min(self._elapsed * 2.0, 1.0)
        total_turn = np.clip(turn + pivot, -1.0, 1.0)

        angles = self._cpg.step(dt, speed=rev_speed, turn=total_turn)
        return angles

    def get_adhesion_states(self) -> np.ndarray:
        return self._cpg.get_adhesion_states()

    def reset(self) -> None:
        self._cpg.reset()
        self._elapsed = 0.0
        self._pivot_dir = 0.0


# ---------------------------------------------------------------------------
# Behaviour controller — brain-driven behaviour switching
# ---------------------------------------------------------------------------


class BehaviorController:
    """Selects the active behaviour based on brain neuron group activity.

    Seven pools of neurons in the brain are monitored:

    * **Walking pool** — high activity → keep walking
    * **Grooming pool** — high activity → antenna grooming
    * **Feeding pool** — high activity → food seeking
    * **Olfactory pool** — high activity → odor search (chemotaxis)
    * **Escape pool** — high activity → startle escape run
    * **Freezing pool** — high activity → tonic immobility
    * **Backward pool** — high activity → reverse and pivot

    The pool with the highest normalised firing rate wins.  A minimum
    hold duration prevents rapid flickering between behaviours.

    Args:
        brain: The :class:`OnlineBrainModel`.
        connectome: The :class:`Connectome`.
        walking_ids: FlyWire IDs for walking-related neurons.
        grooming_ids: FlyWire IDs for grooming-related neurons.
        feeding_ids: FlyWire IDs for feeding-related neurons.
        olfactory_ids: FlyWire IDs for olfaction-related neurons.
        escape_ids: FlyWire IDs for escape/startle neurons.
        freezing_ids: FlyWire IDs for freezing/immobility neurons.
        backward_ids: FlyWire IDs for backward walking neurons.
        hold_time_s: Minimum time (seconds) to stay in a behaviour.
        grooming_threshold: Minimum rate (Hz) for grooming to activate.
        feeding_threshold: Minimum rate (Hz) for feeding to activate.
    """

    def __init__(
        self,
        brain: "OnlineBrainModel",
        connectome: "Connectome",
        walking_ids: list[int],
        grooming_ids: list[int],
        feeding_ids: list[int],
        olfactory_ids: list[int] | None = None,
        escape_ids: list[int] | None = None,
        freezing_ids: list[int] | None = None,
        backward_ids: list[int] | None = None,
        *,
        hold_time_s: float = 1.0,
        grooming_threshold: float = 0.8,
        feeding_threshold: float = 0.6,
    ) -> None:
        self.brain = brain
        self.connectome = connectome
        self.hold_time_s = hold_time_s
        self.grooming_threshold = grooming_threshold
        self.feeding_threshold = feeding_threshold

        # Map FlyWire IDs → local indices in the brain model
        self._walking_local = self._resolve_ids(walking_ids)
        self._grooming_local = self._resolve_ids(grooming_ids)
        self._feeding_local = self._resolve_ids(feeding_ids)
        self._olfactory_local = self._resolve_ids(olfactory_ids or [])
        self._escape_local = self._resolve_ids(escape_ids or [])
        self._freezing_local = self._resolve_ids(freezing_ids or [])
        self._backward_local = self._resolve_ids(backward_ids or [])

        self._state = BehaviorState.WALKING
        self._state_timer: float = 0.0
        self._current_event: str = "baseline"

    def _resolve_ids(self, fly_ids: list[int]) -> list[int]:
        """Convert FlyWire IDs to indices in the motor_rates array.

        The motor_rates array from ``brain.get_motor_spike_rates()`` has
        one entry per motor neuron, in the same order they were passed to
        ``OnlineBrainModel(motor_neuron_ids=...)``.  We find which slots
        in that array correspond to the given FlyWire IDs.
        """
        # _motor_global holds the *global connectome indices* of the motor
        # neurons, in the order they appear in the spike-rate output.
        motor_global = self.brain._motor_global
        # Build flyid → motor-array-slot mapping
        try:
            slot_map = {
                self.connectome.index_to_id(gi): slot
                for slot, gi in enumerate(motor_global)
            }
        except Exception:
            return []

        return [slot_map[fid] for fid in fly_ids if fid in slot_map]

    @property
    def state(self) -> BehaviorState:
        return self._state

    def update(self, dt: float, event: str = "baseline") -> BehaviorState:
        """Evaluate brain activity and return the active behaviour.

        Hybrid approach: the environmental event **gates** which behaviour
        pools are candidates, but the brain must **confirm** the switch
        by showing excess neural activity in the relevant pool above
        the motor-neuron baseline.  This prevents Poisson background
        noise from causing false switches while keeping the brain in
        the decision loop.

        Args:
            dt: Elapsed time since last call (seconds).
            event: Current environment event name.
        """
        self._state_timer += dt
        self._current_event = event

        # Check hold time
        if self._state_timer < self.hold_time_s:
            return self._state

        # --- Compute pool excess over motor baseline ---
        motor_rates = self.brain.get_motor_spike_rates()
        if len(motor_rates) == 0:
            return self._state

        baseline = float(np.mean(motor_rates)) if len(motor_rates) > 0 else 0.0

        groom_excess = max(
            self._pool_rate(motor_rates, self._grooming_local) - baseline, 0.0
        )
        feed_excess = max(
            self._pool_rate(motor_rates, self._feeding_local) - baseline, 0.0
        )
        olfactory_excess = max(
            self._pool_rate(motor_rates, self._olfactory_local) - baseline, 0.0
        )
        escape_excess = max(
            self._pool_rate(motor_rates, self._escape_local) - baseline, 0.0
        )
        freezing_excess = max(
            self._pool_rate(motor_rates, self._freezing_local) - baseline, 0.0
        )
        backward_excess = max(
            self._pool_rate(motor_rates, self._backward_local) - baseline, 0.0
        )

        # --- Hybrid decision ---
        # Event gates the candidate; brain confirms with excess activity.
        # A small minimum (1.0 Hz) prevents switching on pure noise,
        # but is easy to reach when the event injects real current.
        min_confirm_hz = 1.0
        new_state = self._state

        # Escape has highest priority (threat response)
        if event == "vibration_threat" and escape_excess >= min_confirm_hz:
            new_state = BehaviorState.ESCAPE
        elif event == "shadow_overhead" and freezing_excess >= min_confirm_hz:
            new_state = BehaviorState.FREEZING
        elif event == "frontal_collision" and backward_excess >= min_confirm_hz:
            new_state = BehaviorState.BACKWARD
        elif event == "antenna_irritation" and groom_excess >= min_confirm_hz:
            new_state = BehaviorState.GROOMING
        elif event == "sugar_detection" and feed_excess >= min_confirm_hz:
            new_state = BehaviorState.FEEDING
        elif event == "odor_detection" and olfactory_excess >= min_confirm_hz:
            new_state = BehaviorState.SEARCHING
        elif event == "baseline":
            # No event active → return to walking
            new_state = BehaviorState.WALKING

        if new_state != self._state:
            self._state = new_state
            self._state_timer = 0.0

        return self._state

    @staticmethod
    def _pool_rate(rates: np.ndarray, indices: list[int]) -> float:
        if not indices:
            return 0.0
        valid = [i for i in indices if i < len(rates)]
        if not valid:
            return 0.0
        return float(np.mean(rates[valid]))

    def reset(self) -> None:
        self._state = BehaviorState.WALKING
        self._state_timer = 0.0
        self._current_event = "baseline"


# ---------------------------------------------------------------------------
# Sensory event generator — simulated environmental stimuli
# ---------------------------------------------------------------------------


class SensoryEventGenerator:
    """Generates random environmental events that stimulate the brain.

    The real fly lives in a world with unpredictable stimuli: dust on the
    antennae triggers grooming, encountering a sugar patch triggers feeding,
    and quiescent periods allow walking.  This class simulates those events
    by injecting differential currents into specific sensory neuron groups,
    causing the brain to switch behaviours *from its own neural dynamics*
    rather than a fixed schedule.

    Events:
        * **antenna_irritation** — strong excitation of mechano-sensory
          neurons → drives grooming neuron pools
        * **sugar_detection** — strong excitation of gustatory/sugar-sensing
          neurons → drives feeding neuron pools
        * **odor_detection** — moderate excitation of olfactory proxy
          neurons → drives chemotaxis search behaviour
        * **vibration_threat** — sharp burst on mechano-sensory neurons
          → drives escape/startle response
        * **shadow_overhead** — broad inhibitory-like stimulus
          → drives freezing (tonic immobility)
        * **frontal_collision** — front-leg mechanosensors spike
          → drives backward walking / retreat
        * **baseline** — moderate tonic excitation → walking

    Args:
        n_sensory: Total number of sensory neurons.
        event_interval_s: Mean time between events (Poisson process).
        event_duration_s: How long each event persists.
        irritation_strength_mv: Extra current for antenna irritation (mV).
        sugar_strength_mv: Extra current for sugar detection (mV).
        odor_strength_mv: Extra current for odor detection (mV).
        threat_strength_mv: Extra current for vibration threat (mV).
        shadow_strength_mv: Extra current for shadow overhead (mV).
        collision_strength_mv: Extra current for frontal collision (mV).
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        n_sensory: int,
        *,
        event_interval_s: float = 4.0,
        event_duration_s: float = 2.0,
        irritation_strength_mv: float = 25.0,
        sugar_strength_mv: float = 20.0,
        odor_strength_mv: float = 22.0,
        threat_strength_mv: float = 30.0,
        shadow_strength_mv: float = 18.0,
        collision_strength_mv: float = 26.0,
        seed: int | None = None,
    ) -> None:
        self.n_sensory = n_sensory
        self.event_interval_s = event_interval_s
        self.event_duration_s = event_duration_s
        self.irritation_strength_mv = irritation_strength_mv
        self.sugar_strength_mv = sugar_strength_mv
        self.odor_strength_mv = odor_strength_mv
        self.threat_strength_mv = threat_strength_mv
        self.shadow_strength_mv = shadow_strength_mv
        self.collision_strength_mv = collision_strength_mv
        self._rng = np.random.default_rng(seed)

        self._current_event: str = "baseline"
        self._event_timer: float = 0.0
        self._next_event_at: float = self._sample_next_time()

    def _sample_next_time(self) -> float:
        return self._rng.exponential(self.event_interval_s)

    def step(self, dt: float) -> np.ndarray:
        """Advance and return extra sensory currents (mV).

        Returns:
            Shape ``(n_sensory,)`` — extra current on top of baseline.
        """
        extra = np.zeros(self.n_sensory)
        self._event_timer += dt

        # Check if current event has ended
        if self._current_event != "baseline":
            if self._event_timer >= self.event_duration_s:
                self._current_event = "baseline"
                self._event_timer = 0.0
                self._next_event_at = self._sample_next_time()

        # Check if it's time for a new event
        if self._current_event == "baseline":
            if self._event_timer >= self._next_event_at:
                # Pick random event type (7 behaviours)
                r = self._rng.random()
                if r < 0.20:
                    self._current_event = "antenna_irritation"
                elif r < 0.35:
                    self._current_event = "sugar_detection"
                elif r < 0.50:
                    self._current_event = "odor_detection"
                elif r < 0.65:
                    self._current_event = "vibration_threat"
                elif r < 0.77:
                    self._current_event = "shadow_overhead"
                elif r < 0.89:
                    self._current_event = "frontal_collision"
                else:
                    self._current_event = "baseline"  # false alarm
                self._event_timer = 0.0

        # Build stimulus based on current event
        if self._current_event == "antenna_irritation":
            # Stimulate mechano-sensory group (first quarter of sensory neurons)
            group = max(self.n_sensory // 4, 1)
            # Ramp up then plateau
            ramp = min(self._event_timer / 0.3, 1.0)
            extra[:group] = self.irritation_strength_mv * ramp
        elif self._current_event == "sugar_detection":
            # Stimulate gustatory group (second quarter of sensory neurons)
            group = max(self.n_sensory // 4, 1)
            ramp = min(self._event_timer / 0.3, 1.0)
            start = max(self.n_sensory // 4, 1)
            extra[start:start + group] = self.sugar_strength_mv * ramp
        elif self._current_event == "odor_detection":
            # Stimulate olfactory group (last half of sensory neurons)
            group = max(self.n_sensory // 2, 1)
            ramp = min(self._event_timer / 0.5, 1.0)
            fluct = 0.7 + 0.3 * np.sin(self._event_timer * 8.0)
            extra[-group:] = self.odor_strength_mv * ramp * fluct
        elif self._current_event == "vibration_threat":
            # Sharp burst across all sensory neurons (ground vibration)
            # Very fast onset, then sustained at lower level
            burst = np.exp(-self._event_timer * 5.0)  # fast decay
            sustained = 0.4
            strength = max(burst, sustained)
            extra[:] = self.threat_strength_mv * strength
        elif self._current_event == "shadow_overhead":
            # Broad, diffuse stimulus across visual-like neurons
            # (middle third of sensory array)
            group = max(self.n_sensory // 3, 1)
            start = max(self.n_sensory // 3, 1)
            ramp = min(self._event_timer / 0.2, 1.0)
            extra[start:start + group] = self.shadow_strength_mv * ramp
        elif self._current_event == "frontal_collision":
            # Front mechano-sensors spike (first sixth = front legs)
            group = max(self.n_sensory // 6, 1)
            # Sharp onset like hitting a wall
            ramp = min(self._event_timer / 0.1, 1.0)
            extra[:group] = self.collision_strength_mv * ramp

        return extra

    @property
    def current_event(self) -> str:
        return self._current_event

    def reset(self, seed: int | None = None) -> None:
        self._current_event = "baseline"
        self._event_timer = 0.0
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._next_event_at = self._sample_next_time()


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

    def motor_rates_to_leg_commands(
        self,
        motor_rates: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Convert motor rates to per-leg speeds and joint-level modulation.

        Instead of collapsing 114 motor neurons into 2 numbers (speed, turn),
        this maps them to 6 individual leg speeds and per-joint posture
        offsets — giving the brain fine-grained motor control.

        Motor neurons are split into 6 groups (LF, LM, LH, RF, RM, RH).
        Each group's mean firing rate drives its leg's stepping speed.
        The variance within each group creates subtle joint angle offsets
        (the brain modulating posture, not just speed).

        Args:
            motor_rates: Shape ``(n_motor,)`` — firing rates in Hz.

        Returns:
            leg_speeds: Shape ``(6,)`` — per-leg speed multipliers.
            joint_offsets: Shape ``(6, 7)`` — angle offsets per joint (rad).
            turn: Turning signal from left/right rate asymmetry.
        """
        if len(motor_rates) == 0:
            return np.full(6, 0.5), np.zeros((6, 7)), 0.0

        n = len(motor_rates)
        group_size = max(n // 6, 1)

        leg_rates = np.zeros(6)
        joint_offsets = np.zeros((6, 7))

        for i in range(6):
            start = i * group_size
            end = min(start + group_size, n)
            if start >= n:
                leg_rates[i] = np.mean(motor_rates)
                continue
            grp = motor_rates[start:end]
            leg_rates[i] = np.mean(grp)

            # Variance within the group → very subtle joint modulation
            if len(grp) > 1:
                var = np.var(grp)
                # Keep offsets tiny (max ~0.02 rad) and symmetrical
                # so they don't push legs into unnatural poses
                offset = np.tanh(var * 0.5) * 0.02
                # Coxa pitch: slight stride modulation (±)
                joint_offsets[i, 0] += offset * 0.3
                # Femur pitch: slight height modulation (±)
                joint_offsets[i, 3] -= offset * 0.2
                # Tibia pitch: slight foot placement
                joint_offsets[i, 5] += offset * 0.15

        # Normalise rates → speed multipliers (keep legs close together)
        mean_rate = np.mean(motor_rates)
        if mean_rate > 0.1:
            leg_speeds = leg_rates / mean_rate
        else:
            leg_speeds = np.ones(6)
        # Tight range: legs stay between 0.4× and 1.6× mean speed
        leg_speeds = np.clip(0.3 + leg_speeds * self.motor_speed_gain * 200, 0.4, 1.6)

        # Turn from L/R asymmetry (legs 0-2 = left, 3-5 = right)
        left_rate = np.mean(leg_rates[:3])
        right_rate = np.mean(leg_rates[3:])
        turn = np.clip((left_rate - right_rate) * self.motor_turn_gain, -1.0, 1.0)

        return leg_speeds, joint_offsets, turn

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
    # DOF order matches MotionSnippet.dofs_per_leg:
    # (coxa_pitch, coxa_roll, coxa_yaw, femur_pitch, femur_roll, tibia_pitch, tarsus_pitch)
    cpg_dof_spec = WalkingPatternGenerator.DOF_SPEC

    # Build lookup: (leg_pos, parent_link, child_link, axis) → cpg index
    cpg_lookup: dict[tuple[str, str, str, str], tuple[int, int]] = {}
    for leg_i, leg in enumerate(cpg_leg_order):
        for dof_j, (parent, child, axis) in enumerate(cpg_dof_spec):
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
