"""Anatomical Drosophila brain visualizer — real-time neuron activity.

Renders all ~9,000 neurons of the subnetwork as a point cloud shaped
like the dorsal view of the fly brain.  Active neurons glow bright
yellow/white; quiet neurons stay dim purple.  Runs in a separate
process to avoid GUI-thread conflicts with MuJoCo on macOS.
"""

from __future__ import annotations

import multiprocessing as mp
from multiprocessing import Process, Queue
from queue import Empty

import numpy as np


# -----------------------------------------------------------------------
# Brain-shaped point cloud generator
# -----------------------------------------------------------------------

def _generate_brain_positions(n: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Create 2D positions for *n* neurons in a Drosophila brain shape.

    Dorsal view: two optic lobes, central protocerebrum, mushroom bodies,
    and subesophageal zone.

    Returns (x, y) arrays of shape (n,).
    """
    rng = np.random.default_rng(seed)
    pts_x: list[float] = []
    pts_y: list[float] = []

    def _fill_ellipse(cx, cy, rx, ry, count, rng):
        xs, ys = [], []
        batch = count * 4
        while len(xs) < count:
            x = rng.normal(cx, rx * 0.55, batch)
            y = rng.normal(cy, ry * 0.55, batch)
            mask = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 < 1.0
            xs.extend(x[mask].tolist())
            ys.extend(y[mask].tolist())
        return xs[:count], ys[:count]

    # Central brain (protocerebrum)
    x, y = _fill_ellipse(0, 0, 0.42, 0.26, int(n * 0.40), rng)
    pts_x.extend(x); pts_y.extend(y)

    # Left optic lobe
    x, y = _fill_ellipse(-0.62, 0.0, 0.28, 0.30, int(n * 0.22), rng)
    # Slight outward bulge
    for i in range(len(x)):
        x[i] -= 0.05 * np.exp(-y[i] ** 2 / 0.05)
    pts_x.extend(x); pts_y.extend(y)

    # Right optic lobe (mirror)
    rx = [-xi for xi in x]
    pts_x.extend(rx); pts_y.extend(y)

    # Mushroom bodies (two small clusters)
    for sx in (-0.18, 0.18):
        x, y = _fill_ellipse(sx, 0.14, 0.07, 0.08, int(n * 0.03), rng)
        pts_x.extend(x); pts_y.extend(y)

    # Subesophageal zone (ventral)
    x, y = _fill_ellipse(0, -0.28, 0.16, 0.08, int(n * 0.05), rng)
    pts_x.extend(x); pts_y.extend(y)

    # Antennal lobes (two small anterior clusters)
    for sx in (-0.12, 0.12):
        x, y = _fill_ellipse(sx, 0.22, 0.06, 0.05, int(n * 0.02), rng)
        pts_x.extend(x); pts_y.extend(y)

    all_x = np.array(pts_x, dtype=np.float32)
    all_y = np.array(pts_y, dtype=np.float32)

    have = len(all_x)
    if have >= n:
        idx = rng.choice(have, n, replace=False)
        return all_x[idx], all_y[idx]
    else:
        extra = n - have
        idx = rng.choice(have, extra, replace=True)
        jx = all_x[idx] + rng.normal(0, 0.008, extra).astype(np.float32)
        jy = all_y[idx] + rng.normal(0, 0.008, extra).astype(np.float32)
        return np.concatenate([all_x, jx]), np.concatenate([all_y, jy])


# -----------------------------------------------------------------------
# Child-process rendering loop
# -----------------------------------------------------------------------

def _visualizer_loop(
    q: Queue,
    pos_x: np.ndarray,
    pos_y: np.ndarray,
    n_neurons: int,
    motor_local: list[int],
    sensory_local: list[int],
) -> None:
    """Matplotlib rendering loop running in a child process."""

    import matplotlib
    for backend in ("macosx", "qt5agg", "qtagg", "tkagg"):
        try:
            matplotlib.use(backend)
            break
        except Exception:
            continue
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    # Custom colormap: dark purple → magenta → yellow → white
    cmap_colors = [
        (0.06, 0.02, 0.15),   # very dark purple (silent)
        (0.15, 0.04, 0.30),   # dark purple
        (0.45, 0.05, 0.55),   # purple
        (0.80, 0.15, 0.40),   # magenta-pink
        (1.00, 0.55, 0.10),   # orange-yellow
        (1.00, 0.85, 0.30),   # bright yellow
        (1.00, 1.00, 0.85),   # near white
    ]
    brain_cmap = LinearSegmentedColormap.from_list("brain_glow", cmap_colors, N=256)

    BG = "#050510"

    plt.ion()
    fig = plt.figure(figsize=(8, 6), facecolor=BG)
    fig.canvas.manager.set_window_title("Drosophila Brain — Neural Activity")

    ax = fig.add_axes([0.02, 0.08, 0.96, 0.84])
    ax.set_facecolor(BG)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-0.55, 0.55)
    ax.set_aspect("equal")
    ax.axis("off")

    # Title
    ax.text(0, 0.48, "simultaneous brain emulation",
            ha="center", va="bottom", fontsize=11,
            color="#888888", fontstyle="italic")

    # Base dim colors — visible enough to show brain shape
    base_rgba = np.full((n_neurons, 4), [0.28, 0.10, 0.45, 0.55],
                        dtype=np.float32)
    motor_set = set(motor_local)
    sensory_set = set(sensory_local)
    for idx in sensory_set:
        base_rgba[idx] = [0.20, 0.40, 0.65, 0.60]
    for idx in motor_set:
        base_rgba[idx] = [0.35, 0.12, 0.50, 0.60]

    # Glow layer (large soft halos behind active neurons)
    glow_scatter = ax.scatter(
        pos_x, pos_y, s=3.0,
        c=np.zeros((n_neurons, 4)),
        edgecolors="none", zorder=1,
    )

    # Main neuron scatter — base size visible
    main_scatter = ax.scatter(
        pos_x, pos_y, s=6.0,
        c=base_rgba.copy(),
        edgecolors="none", zorder=2,
    )

    # Info text
    info_txt = ax.text(
        0, -0.48, "", ha="center", va="top", fontsize=10,
        color="#aaaaaa", fontweight="bold",
    )
    count_txt = ax.text(
        -0.95, -0.48, "", ha="left", va="top", fontsize=8,
        color="#666666",
    )

    fig.canvas.draw()
    fig.canvas.flush_events()

    # Preallocate
    main_rgba = np.zeros((n_neurons, 4), dtype=np.float32)
    glow_rgba = np.zeros((n_neurons, 4), dtype=np.float32)

    beh_colors = {"walking": "#4FC3F7", "grooming": "#66BB6A",
                  "feeding": "#FF7043"}
    evt_labels = {"baseline": "", "antenna_irritation": "  •  antenna stimulus",
                  "sugar_detection": "  •  sugar detected"}

    while True:
        if not plt.fignum_exists(fig.number):
            break

        data = None
        try:
            while True:
                data = q.get_nowait()
        except Empty:
            pass

        if data is None:
            plt.pause(0.03)
            continue
        if data == "STOP":
            break

        rates = data["rates"]
        event = data["event"]
        behavior = data["behavior"]

        max_rate = max(float(np.max(rates)), 1.0)
        norm = np.clip(rates / max_rate, 0, 1).astype(np.float32)

        active_mask = norm > 0.005
        mapped = brain_cmap(norm)  # (n, 4)

        # Main scatter colors — vectorized
        main_rgba[:] = base_rgba
        alphas = np.clip(0.5 + norm * 0.5, 0, 1)
        main_rgba[active_mask, :3] = mapped[active_mask, :3]
        main_rgba[active_mask, 3] = alphas[active_mask]

        sizes = np.where(active_mask, 6.0 + norm * 25.0, 6.0)
        main_scatter.set_facecolors(main_rgba)
        main_scatter.set_sizes(sizes)

        # Glow halos — vectorized
        glow_rgba[:] = 0
        glow_mask = norm > 0.05
        glow_rgba[glow_mask, :3] = mapped[glow_mask, :3]
        glow_rgba[glow_mask, 3] = norm[glow_mask] * 0.35
        glow_sizes = np.where(glow_mask, 20.0 + norm * 70.0, 0.0)
        glow_scatter.set_facecolors(glow_rgba)
        glow_scatter.set_sizes(glow_sizes)

        # Text
        n_active = int(np.sum(active_mask))
        beh_col = beh_colors.get(behavior, "#aaaaaa")
        info_txt.set_text(
            f"{behavior.upper()}{evt_labels.get(event, '')}")
        info_txt.set_color(beh_col)
        count_txt.set_text(f"{n_active:,} / {n_neurons:,} active")

        try:
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
        except Exception:
            break

        plt.pause(0.02)

    plt.close(fig)


# -----------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------

class BrainVisualizer:
    """Anatomical brain activity viewer in a separate process.

    Shows all neurons in the subnetwork as a 2D point cloud shaped
    like the dorsal view of the Drosophila brain.  Active neurons
    glow from purple to yellow to white.

    Args:
        brain: :class:`OnlineBrainModel` instance.
        beh_ctrl: :class:`BehaviorController` instance.
        update_every: Send data every N calls to :meth:`update`.
    """

    def __init__(self, brain, beh_ctrl, *, update_every: int = 2) -> None:
        self.brain = brain
        self.beh_ctrl = beh_ctrl
        self.update_every = update_every
        self._call_count = 0

        n = brain.n_neurons
        self._n_neurons = n

        # Generate brain-shaped positions
        pos_x, pos_y = _generate_brain_positions(n)

        self._queue: Queue = Queue(maxsize=30)
        self._process = Process(
            target=_visualizer_loop,
            args=(
                self._queue,
                pos_x,
                pos_y,
                n,
                list(brain._motor_local),
                list(brain._sensory_local),
            ),
            daemon=True,
        )
        self._process.start()

    def update(self, current_event: str, current_behavior) -> None:
        """Send latest brain state to the visualizer."""
        self._call_count += 1
        if self._call_count % self.update_every != 0:
            return
        if not self._process.is_alive():
            return

        rates = self.brain.get_all_spike_rates_array()

        try:
            self._queue.put_nowait({
                "rates": rates,
                "event": current_event,
                "behavior": current_behavior.value,
            })
        except Exception:
            pass

    def close(self) -> None:
        """Shut down the visualizer process."""
        try:
            self._queue.put_nowait("STOP")
        except Exception:
            pass
        self._process.join(timeout=3)
        if self._process.is_alive():
            self._process.terminate()
