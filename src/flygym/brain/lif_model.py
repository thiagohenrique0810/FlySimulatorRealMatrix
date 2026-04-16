"""Online (step-by-step) LIF brain model for closed-loop simulation.

Wraps the Brian2 spiking neural network from the Drosophila brain model
so it can be advanced incrementally and coupled with FlyGym physics.
"""

from __future__ import annotations

from collections import deque
from textwrap import dedent
from typing import Sequence

import numpy as np

from flygym.brain.connectome import Connectome


# ---------------------------------------------------------------------------
# Default LIF parameters (from Shiu et al.)
# ---------------------------------------------------------------------------

DEFAULT_LIF_PARAMS: dict = {
    "v_0": -52.0,        # resting potential (mV)
    "v_rst": -52.0,      # reset potential (mV)
    "v_th": -45.0,       # spiking threshold (mV)
    "t_mbr": 20.0,       # membrane time constant (ms)
    "tau": 5.0,          # synaptic time constant (ms)
    "t_rfc": 2.2,        # refractory period (ms)
    "t_dly": 1.8,        # synaptic delay (ms)
    "w_syn": 0.275,      # base synaptic weight (mV)
    "f_poi": 250,        # Poisson weight scaling factor
    "r_poi": 150.0,      # default Poisson rate (Hz) for excited neurons
}


class OnlineBrainModel:
    """Step-by-step Brian2 whole-brain LIF simulation.

    The model creates a ``NeuronGroup`` from the connectome and adds an
    external current variable ``I_ext`` that can be set from Python at each
    brain step to represent sensory input.

    Args:
        connectome: A loaded :class:`Connectome` instance.
        params: LIF parameters overriding :data:`DEFAULT_LIF_PARAMS`.
        sensory_neuron_ids: FlyWire IDs of neurons that receive sensory input.
        motor_neuron_ids: FlyWire IDs of neurons whose output drives the body.
        dt_ms: Internal Brian2 timestep in milliseconds.
        use_subnetwork: If True, build a subnetwork containing only the
            sensory and motor neurons plus their direct pre/post-synaptic
            partners.  Much faster but less biologically accurate.
        spike_window_ms: Width of the sliding window (ms) used to compute
            instantaneous firing rates.
    """

    def __init__(
        self,
        connectome: Connectome,
        *,
        params: dict | None = None,
        sensory_neuron_ids: Sequence[int] | None = None,
        motor_neuron_ids: Sequence[int] | None = None,
        dt_ms: float = 0.1,
        use_subnetwork: bool = False,
        spike_window_ms: float = 50.0,
    ) -> None:
        self.connectome = connectome
        self.params = {**DEFAULT_LIF_PARAMS, **(params or {})}
        self.dt_ms = dt_ms
        self.use_subnetwork = use_subnetwork
        self.spike_window_ms = spike_window_ms

        # Convert FlyWire IDs → global indices
        self._sensory_global = (
            connectome.ids_to_indices(sensory_neuron_ids)
            if sensory_neuron_ids
            else []
        )
        self._motor_global = (
            connectome.ids_to_indices(motor_neuron_ids) if motor_neuron_ids else []
        )

        # Will be filled by setup()
        self._net = None  # Brian2 Network
        self._neu = None  # NeuronGroup
        self._spk_mon = None  # SpikeMonitor
        self._n_neurons: int = 0
        self._global_to_local: dict[int, int] | None = None
        self._sensory_local: list[int] = []
        self._motor_local: list[int] = []

        # Spike history ring buffer for rate computation
        self._spike_history: deque = deque()
        self._current_time_ms: float = 0.0

        self._brian2_imported = False

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self) -> None:
        """Build the Brian2 network.  Call once before stepping."""
        try:
            from brian2 import (
                NeuronGroup,
                Synapses,
                SpikeMonitor,
                Network,
                mV,
                ms,
                Hz,
                defaultclock,
            )
        except ImportError as exc:
            raise ImportError(
                "Brian2 is required for the online brain model.  "
                "Install with:  pip install brian2"
            ) from exc

        self._brian2_imported = True
        p = self.params

        # Set Brian2 timestep
        defaultclock.dt = self.dt_ms * ms

        # --- Determine which neurons to include ---
        if self.use_subnetwork and (self._sensory_global or self._motor_global):
            seed_indices = set(self._sensory_global + self._motor_global)
            # Add direct neighbours (1-hop)
            pre = self.connectome.pre_indices
            post = self.connectome.post_indices
            neighbours: set[int] = set()
            for idx in seed_indices:
                # Post-synaptic targets of this neuron
                mask_pre = pre == idx
                neighbours.update(int(x) for x in post[mask_pre])
                # Pre-synaptic sources to this neuron
                mask_post = post == idx
                neighbours.update(int(x) for x in pre[mask_post])
            all_indices = sorted(seed_indices | neighbours)
            sub_pre, sub_post, sub_w, g2l = self.connectome.extract_subnetwork(
                all_indices
            )
            self._n_neurons = len(all_indices)
            self._global_to_local = g2l
            self._sensory_local = [g2l[g] for g in self._sensory_global if g in g2l]
            self._motor_local = [g2l[g] for g in self._motor_global if g in g2l]
        else:
            # Full network
            self._n_neurons = self.connectome.n_neurons
            self._global_to_local = None
            self._sensory_local = list(self._sensory_global)
            self._motor_local = list(self._motor_global)
            sub_pre = self.connectome.pre_indices
            sub_post = self.connectome.post_indices
            sub_w = self.connectome.synapse_weights

        # --- Create neuron group with external current ---
        eqs = dedent("""
            dv/dt = (v_0 - v + g + I_ext) / t_mbr : volt (unless refractory)
            dg/dt = -g / tau : volt (unless refractory)
            I_ext : volt
            rfc : second
        """)

        neu = NeuronGroup(
            N=self._n_neurons,
            model=eqs,
            method="linear",
            threshold="v > v_th",
            reset="v = v_rst; g = 0 * mV",
            refractory="rfc",
            name="brain_neurons",
            namespace={
                "v_0": p["v_0"] * mV,
                "v_rst": p["v_rst"] * mV,
                "v_th": p["v_th"] * mV,
                "t_mbr": p["t_mbr"] * ms,
                "tau": p["tau"] * ms,
            },
        )
        neu.v = p["v_0"] * mV
        neu.g = 0 * mV
        neu.I_ext = 0 * mV
        neu.rfc = p["t_rfc"] * ms

        # Sensory neurons: remove refractory period so they respond immediately
        for idx in self._sensory_local:
            neu[idx].rfc = 0 * ms

        # --- Create synapses ---
        syn = Synapses(
            neu,
            neu,
            "w : volt",
            on_pre="g += w",
            delay=p["t_dly"] * ms,
            name="brain_synapses",
        )
        if len(sub_pre) > 0:
            syn.connect(i=sub_pre, j=sub_post)
            syn.w = sub_w.astype(np.float64) * p["w_syn"] * mV

        # --- Spike monitor ---
        spk_mon = SpikeMonitor(neu)

        # --- Assemble network ---
        self._net = Network(neu, syn, spk_mon)
        self._neu = neu
        self._spk_mon = spk_mon
        self._spike_history.clear()
        self._current_time_ms = 0.0

    # ------------------------------------------------------------------
    # Runtime
    # ------------------------------------------------------------------

    def step(self, duration_ms: float) -> None:
        """Advance the brain simulation by *duration_ms* milliseconds."""
        if self._net is None:
            raise RuntimeError("Call setup() before stepping the brain model.")
        from brian2 import ms as ms_unit

        self._net.run(duration_ms * ms_unit)
        self._current_time_ms += duration_ms

        # Record spikes for rate computation
        trains = self._spk_mon.spike_trains()
        t_start = (self._current_time_ms - duration_ms) / 1000.0  # seconds
        for neu_idx, times in trains.items():
            for t in times:
                t_sec = float(t)
                if t_sec >= t_start:
                    self._spike_history.append((int(neu_idx), t_sec))

        # Prune old spikes outside the rate window
        cutoff_sec = (self._current_time_ms - self.spike_window_ms) / 1000.0
        while self._spike_history and self._spike_history[0][1] < cutoff_sec:
            self._spike_history.popleft()

    def inject_sensory_current(
        self,
        currents_mv: np.ndarray,
    ) -> None:
        """Set external currents on sensory neurons.

        Args:
            currents_mv: Array of shape ``(n_sensory,)`` with currents in mV.
                Positive values depolarise; negative values hyperpolarise.
        """
        if self._neu is None:
            raise RuntimeError("Call setup() first.")
        from brian2 import mV

        if len(currents_mv) != len(self._sensory_local):
            raise ValueError(
                f"Expected {len(self._sensory_local)} currents, "
                f"got {len(currents_mv)}"
            )
        # Reset all external currents to zero
        self._neu.I_ext = 0 * mV
        # Set sensory neuron currents
        for local_idx, current in zip(self._sensory_local, currents_mv):
            self._neu[local_idx].I_ext = current * mV

    def get_motor_spike_rates(self) -> np.ndarray:
        """Compute instantaneous firing rates (Hz) for motor neurons.

        Uses a sliding window of width ``spike_window_ms``.

        Returns:
            Array of shape ``(n_motor_neurons,)`` with rates in Hz.
        """
        window_sec = self.spike_window_ms / 1000.0
        counts = np.zeros(len(self._motor_local), dtype=np.float64)
        local_set = {loc: i for i, loc in enumerate(self._motor_local)}

        for neu_idx, t_sec in self._spike_history:
            if neu_idx in local_set:
                counts[local_set[neu_idx]] += 1

        rates = counts / max(window_sec, 1e-9)
        return rates

    def get_all_spike_rates(self) -> dict[int, float]:
        """Compute firing rates (Hz) for all neurons that spiked in the window.

        Returns:
            Dict mapping local neuron index to firing rate in Hz.
        """
        window_sec = self.spike_window_ms / 1000.0
        counts: dict[int, int] = {}
        for neu_idx, _ in self._spike_history:
            counts[neu_idx] = counts.get(neu_idx, 0) + 1
        return {k: v / max(window_sec, 1e-9) for k, v in counts.items()}

    def reset(self) -> None:
        """Reset the brain model to its initial state."""
        if self._net is None:
            return
        from brian2 import mV, ms

        p = self.params
        self._neu.v = p["v_0"] * mV
        self._neu.g = 0 * mV
        self._neu.I_ext = 0 * mV
        self._spike_history.clear()
        self._current_time_ms = 0.0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_neurons(self) -> int:
        return self._n_neurons

    @property
    def n_sensory(self) -> int:
        return len(self._sensory_local)

    @property
    def n_motor(self) -> int:
        return len(self._motor_local)

    @property
    def current_time_ms(self) -> float:
        return self._current_time_ms


# ---------------------------------------------------------------------------
# Batch runner (uses original model.py for full experiments)
# ---------------------------------------------------------------------------


def run_batch_experiment(
    connectome: Connectome,
    experiment_name: str,
    excited_neuron_ids: Sequence[int],
    *,
    silenced_neuron_ids: Sequence[int] = (),
    results_dir: str = "./results",
    params: dict | None = None,
    n_proc: int = -1,
    force_overwrite: bool = False,
) -> None:
    """Run a full batch experiment using the original Shiu et al. model.

    This calls ``run_exp()`` from the cloned Drosophila_brain_model repository,
    which runs multiple trials in parallel and stores spike data to disk.

    Args:
        connectome: Loaded connectome instance.
        experiment_name: Unique name for this experiment.
        excited_neuron_ids: FlyWire IDs of neurons to excite (Poisson input).
        silenced_neuron_ids: FlyWire IDs of neurons to silence.
        results_dir: Directory where results are stored.
        params: LIF parameters (uses defaults if None).
        n_proc: Number of CPU cores (-1 = all).
        force_overwrite: Overwrite existing results.
    """
    import sys
    from pathlib import Path

    # Add the cloned repo to sys.path so we can import model.py
    brain_repo = connectome.data_dir
    if str(brain_repo) not in sys.path:
        sys.path.insert(0, str(brain_repo))

    from model import run_exp, default_params  # type: ignore[import-not-found]

    merged_params = {**default_params}
    if params:
        from brian2 import mV, ms, Hz

        for k, v in params.items():
            if k in ("v_0", "v_rst", "v_th", "w_syn"):
                merged_params[k] = v * mV
            elif k in ("t_mbr", "tau", "t_rfc", "t_dly"):
                merged_params[k] = v * ms
            elif k in ("r_poi", "r_poi2"):
                merged_params[k] = v * Hz
            else:
                merged_params[k] = v

    run_exp(
        exp_name=experiment_name,
        neu_exc=list(excited_neuron_ids),
        path_res=results_dir,
        path_comp=str(connectome.completeness_path),
        path_con=str(connectome.connectivity_path),
        params=merged_params,
        neu_slnc=list(silenced_neuron_ids),
        n_proc=n_proc,
        force_overwrite=force_overwrite,
    )
