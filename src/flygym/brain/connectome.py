"""Load and query the FlyWire connectome data.

Wraps the Completeness and Connectivity data from the Drosophila brain model
(philshiu/Drosophila_brain_model) for use with FlyGym.
"""

from __future__ import annotations

import csv
import pickle
from pathlib import Path
from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Known neuron IDs for specific behaviours
# ---------------------------------------------------------------------------

# Sugar‑sensing gustatory receptor neurons (right hemisphere)
SUGAR_SENSING_RIGHT_IDS: list[int] = [
    720575940617937543,
    720575940621754367,
    720575940622695448,
    720575940625815264,
    720575940613236464,
    720575940613828302,
    720575940616553449,
    720575940620078329,
    720575940620558198,
    720575940622002840,
    720575940624539778,
    720575940625362705,
    720575940625815264,
    720575940626010498,
    720575940626055890,
    720575940629257580,
    720575940631116261,
    720575940632524076,
    720575940642181113,
    720575940657491046,
    720575940660219265,
]

# Motor neuron 9 (MN9) – leg motor neuron
MN9_FLYWIRE_ID: int = 720575940660219265

# SEZ types associated with each behaviour
WALKING_SEZ_TYPES: list[str] = [
    "rocket", "basket", "horseshoe", "diatom", "mime",
    "weaver", "gallinule", "mandala", "brontosaraus", "oink",
]

GROOMING_SEZ_TYPES: list[str] = [
    "broom",      # 2 neurons – name implies grooming
    "whisker",    # 2 neurons – sensory / grooming‑like
    "handle",     # 4 neurons – manipulative limb motion
    "earmuff",    # 4 neurons – head‑area related
]

FEEDING_SEZ_TYPES: list[str] = [
    "Fdg",            # 2 neurons – likely "feeding"
    "Salivary_MN13",  # 2 neurons – salivary motor neuron
    "TPN4",           # 2 neurons – taste projection neuron
    "FMIn",           # 2 neurons – feeding motor interneuron
]

# Olfactory-related SEZ types (flower-named neuron types —
# repurposed as olfaction-to-motor relay for odor search behaviour)
OLFACTORY_SEZ_TYPES: list[str] = [
    "hyacinth",   # 6 neurons – flower-named, sensory integration
    "foxglove",   # 2 neurons – flower-named
    "tulip",      # 2 neurons – flower-named
    "rose",       # 1 neuron  – flower-named
    "bluebell",   # 2 neurons – flower-named
    "peacock",    # 3 neurons – sensory integration
]

# Escape / startle response — threat-related SEZ types
# (predator-like names → fight-or-flight motor relay)
ESCAPE_SEZ_TYPES: list[str] = [
    "shark",      # 3 neurons – predator-named
    "lion",       # 2 neurons – predator-named
    "horn",       # 3 neurons – defensive structure
    "snake",      # 2 neurons – threat-named
    "trident",    # 1 neuron  – weapon-shaped
]

# Freezing / immobility — anti-predator response SEZ types
FREEZING_SEZ_TYPES: list[str] = [
    "mute",       # 1 neuron  – silence/inhibition
    "phantom",    # 2 neurons – unseen/still
    "spirit",     # 2 neurons – motionless presence
]

# Backward walking / retreat — obstacle avoidance SEZ types
BACKWARD_SEZ_TYPES: list[str] = [
    "bridle",     # 4 neurons – reining/pulling back
    "rattle",     # 2 neurons – warning/retreat
    "roundup",    # 2 neurons – corralling/reverse
]

# Proxy olfactory receptor neurons — sensory-like neurons
# (very few inputs, many outputs) that serve as stand-ins for
# ORNs since the connectome lacks cell-type annotations.
# Selected from neurons with ≤3 inputs and ≥47 outputs
# (same connectivity profile as real ORNs).
OLFACTORY_PROXY_IDS: list[int] = [
    720575940612522646,
    720575940619003969,
    720575940633284276,
    720575940615110450,
    720575940621765813,
    720575940621708981,
    720575940637321847,
    720575940624822003,
    720575940608366722,
    720575940622097959,
    720575940612857055,
    720575940622051393,
    720575940606303755,
    720575940623405948,
    720575940611268266,
    720575940622674504,
    720575940625865724,
    720575940635697380,
    720575940629561400,
    720575940625429647,
]


class Connectome:
    """Interface to the FlyWire v783 connectome.

    Loads neuron IDs from *Completeness_783.csv* and (optionally) the SEZ
    neuron type mapping from *sez_neurons.pickle*.  Connectivity data is
    loaded lazily when first needed (parquet requires pandas + pyarrow).

    Args:
        data_dir: Directory containing the connectome data files.
            Expected files: ``Completeness_783.csv``,
            ``Connectivity_783.parquet``, ``sez_neurons.pickle``.
    """

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)
        self._completeness_path = self.data_dir / "Completeness_783.csv"
        self._connectivity_path = self.data_dir / "Connectivity_783.parquet"
        self._sez_path = self.data_dir / "sez_neurons.pickle"

        # Load neuron IDs (lightweight — plain CSV)
        self._flywire_ids: list[int] = []
        with open(self._completeness_path, newline="") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                self._flywire_ids.append(int(row[0]))

        # Build bidirectional index
        self._flyid_to_idx: dict[int, int] = {
            fid: idx for idx, fid in enumerate(self._flywire_ids)
        }

        # SEZ neuron types (optional — file may not exist)
        self._sez_types: dict[str, list[int]] | None = None
        if self._sez_path.exists():
            with open(self._sez_path, "rb") as f:
                self._sez_types = pickle.load(f)

        # Connectivity loaded lazily
        self._pre_indices: np.ndarray | None = None
        self._post_indices: np.ndarray | None = None
        self._weights: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_neurons(self) -> int:
        """Total number of neurons in the connectome."""
        return len(self._flywire_ids)

    @property
    def flywire_ids(self) -> list[int]:
        """All FlyWire neuron IDs, in index order."""
        return list(self._flywire_ids)

    @property
    def sez_neuron_types(self) -> dict[str, list[int]]:
        """Mapping from SEZ neuron type name to FlyWire IDs.

        Returns an empty dict if *sez_neurons.pickle* was not found.
        """
        return dict(self._sez_types) if self._sez_types else {}

    @property
    def completeness_path(self) -> Path:
        return self._completeness_path

    @property
    def connectivity_path(self) -> Path:
        return self._connectivity_path

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    def id_to_index(self, flywire_id: int) -> int:
        """Convert a FlyWire ID to its integer index (Brian2 neuron index)."""
        return self._flyid_to_idx[flywire_id]

    def index_to_id(self, index: int) -> int:
        """Convert an integer index to its FlyWire ID."""
        return self._flywire_ids[index]

    def ids_to_indices(self, flywire_ids: Sequence[int]) -> list[int]:
        """Convert a sequence of FlyWire IDs to integer indices."""
        return [self._flyid_to_idx[fid] for fid in flywire_ids]

    def indices_to_ids(self, indices: Sequence[int]) -> list[int]:
        """Convert a sequence of integer indices to FlyWire IDs."""
        return [self._flywire_ids[i] for i in indices]

    # ------------------------------------------------------------------
    # SEZ helpers
    # ------------------------------------------------------------------

    def get_sez_indices(self, type_name: str) -> list[int]:
        """Get Brian2 indices for an SEZ neuron type.

        Args:
            type_name: A key in ``sez_neuron_types`` (e.g. ``"rocket"``).

        Returns:
            List of integer indices in the neuron array.
        """
        if self._sez_types is None:
            raise FileNotFoundError("sez_neurons.pickle not found")
        ids = self._sez_types[type_name]
        return self.ids_to_indices(ids)

    def get_all_sez_indices(self) -> list[int]:
        """Get Brian2 indices for *all* SEZ neurons, deduplicating."""
        if self._sez_types is None:
            return []
        seen: set[int] = set()
        result: list[int] = []
        for ids in self._sez_types.values():
            for fid in ids:
                idx = self._flyid_to_idx.get(fid)
                if idx is not None and idx not in seen:
                    seen.add(idx)
                    result.append(idx)
        return sorted(result)

    # ------------------------------------------------------------------
    # Connectivity (lazy-loaded)
    # ------------------------------------------------------------------

    def _ensure_connectivity_loaded(self) -> None:
        if self._pre_indices is not None:
            return
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "pandas and pyarrow are required to load connectivity data.  "
                "Install them with:  pip install pandas pyarrow"
            ) from exc

        df = pd.read_parquet(self._connectivity_path)
        self._pre_indices = df["Presynaptic_Index"].values.astype(np.int32)
        self._post_indices = df["Postsynaptic_Index"].values.astype(np.int32)
        self._weights = df["Excitatory x Connectivity"].values.astype(np.float32)

    @property
    def pre_indices(self) -> np.ndarray:
        """Presynaptic neuron indices for every synapse."""
        self._ensure_connectivity_loaded()
        return self._pre_indices  # type: ignore[return-value]

    @property
    def post_indices(self) -> np.ndarray:
        """Postsynaptic neuron indices for every synapse."""
        self._ensure_connectivity_loaded()
        return self._post_indices  # type: ignore[return-value]

    @property
    def synapse_weights(self) -> np.ndarray:
        """Signed synaptic weights (Excitatory x Connectivity)."""
        self._ensure_connectivity_loaded()
        return self._weights  # type: ignore[return-value]

    @property
    def n_synapses(self) -> int:
        self._ensure_connectivity_loaded()
        return len(self._pre_indices)  # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Subnetwork extraction
    # ------------------------------------------------------------------

    def extract_subnetwork(
        self,
        neuron_indices: Sequence[int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, int]]:
        """Extract a subnetwork containing only the given neurons.

        Args:
            neuron_indices: Global indices of neurons to keep.

        Returns:
            A tuple ``(sub_pre, sub_post, sub_weights, global_to_local)`` where
            indices are remapped to ``[0, len(neuron_indices))``.
        """
        self._ensure_connectivity_loaded()
        keep = set(neuron_indices)
        global_to_local = {g: i for i, g in enumerate(sorted(keep))}

        mask = np.array(
            [
                (int(p) in keep and int(q) in keep)
                for p, q in zip(self._pre_indices, self._post_indices)  # type: ignore
            ],
            dtype=bool,
        )
        sub_pre = np.array(
            [global_to_local[int(p)] for p in self._pre_indices[mask]],  # type: ignore
            dtype=np.int32,
        )
        sub_post = np.array(
            [global_to_local[int(q)] for q in self._post_indices[mask]],  # type: ignore
            dtype=np.int32,
        )
        sub_w = self._weights[mask].copy()  # type: ignore
        return sub_pre, sub_post, sub_w, global_to_local
