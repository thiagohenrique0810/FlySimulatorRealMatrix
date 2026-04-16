"""Nature-themed world with grass, trees, and food for brain-driven demos."""

from __future__ import annotations

import math
import numpy as np
from typing import override

from flygym.compose.world import FlatGroundWorld


class NatureWorld(FlatGroundWorld):
    """FlatGroundWorld with a natural environment: grass ground, blue sky,
    decorative trees/plants, and food droplets.

    All decorative elements are visual-only (no physics collisions).

    Args:
        name: Name of the world.
        half_size: Half-size of the ground plane in mm.
        n_trees: Number of trees to scatter around.
        n_grass_patches: Number of grass blade clusters.
        n_food_drops: Number of food droplets on the ground.
        n_odor_sources: Number of odor-emitting flowers.
        n_obstacles: Number of small obstacles the fly can bump into.
        food_radius: Radius of food droplets (mm).
        tree_distance_range: (min, max) distance from origin for trees (mm).
        seed: Random seed for placement.
    """

    @override
    def __init__(
        self,
        name: str = "nature_world",
        *,
        half_size: float = 1000,
        n_trees: int = 12,
        n_grass_patches: int = 60,
        n_food_drops: int = 5,
        n_odor_sources: int = 4,
        n_obstacles: int = 6,
        food_radius: float = 0.8,
        tree_distance_range: tuple[float, float] = (15.0, 80.0),
        seed: int = 42,
    ) -> None:
        super().__init__(name=name, half_size=half_size)

        rng = np.random.default_rng(seed)
        root = self.mjcf_root

        # --- Replace checker ground with grass texture ---
        # Remove old checker texture/material references by overriding
        grass_tex = root.asset.add(
            "texture",
            name="grass_tex",
            type="2d",
            builtin="flat",
            width=512,
            height=512,
            rgb1=(0.22, 0.45, 0.12),   # dark green
            rgb2=(0.30, 0.55, 0.18),    # lighter green
            mark="random",
            markrgb=(0.18, 0.38, 0.08),
            random=0.3,
        )
        grass_mat = root.asset.add(
            "material",
            name="grass_mat",
            texture=grass_tex,
            texrepeat=(80, 80),
            reflectance=0.05,
        )
        # Update ground geom to use grass
        self.ground_geom.material = grass_mat

        # --- Skybox ---
        root.asset.add(
            "texture",
            name="sky_tex",
            type="skybox",
            builtin="gradient",
            width=512,
            height=512,
            rgb1=(0.4, 0.6, 0.9),      # light blue
            rgb2=(0.85, 0.92, 1.0),     # pale horizon
        )

        # --- Sunlight ---
        root.worldbody.add(
            "light",
            name="sun",
            pos=(0, 0, 500),
            dir=(0.3, 0.2, -1),
            diffuse=(1.0, 0.95, 0.8),
            specular=(0.3, 0.3, 0.3),
            ambient=(0.15, 0.15, 0.1),
            directional="true",
            castshadow="true",
        )

        # --- Materials for decorations ---
        root.asset.add(
            "material", name="bark_mat",
            rgba=(0.35, 0.22, 0.10, 1),
            reflectance=0.02,
        )
        root.asset.add(
            "material", name="leaf_mat",
            rgba=(0.15, 0.50, 0.10, 0.9),
            reflectance=0.05,
        )
        root.asset.add(
            "material", name="food_mat",
            rgba=(0.95, 0.75, 0.10, 0.85),
            reflectance=0.3,
            emission=0.3,
        )
        root.asset.add(
            "material", name="grass_blade_mat",
            rgba=(0.20, 0.52, 0.15, 0.8),
            reflectance=0.02,
        )
        root.asset.add(
            "material", name="flower_petal_mat",
            rgba=(0.85, 0.35, 0.65, 0.9),
            reflectance=0.1,
            emission=0.15,
        )
        root.asset.add(
            "material", name="flower_center_mat",
            rgba=(1.0, 0.85, 0.2, 1.0),
            reflectance=0.15,
            emission=0.4,
        )

        # --- Trees (trunk cylinder + canopy ellipsoid) ---
        for i in range(n_trees):
            dist = rng.uniform(*tree_distance_range)
            angle = rng.uniform(0, 2 * math.pi)
            x = dist * math.cos(angle)
            y = dist * math.sin(angle)

            trunk_h = rng.uniform(8.0, 18.0)
            trunk_r = rng.uniform(0.4, 0.9)
            canopy_r = rng.uniform(3.0, 7.0)

            # Trunk
            root.worldbody.add(
                "geom",
                name=f"tree_trunk_{i}",
                type="cylinder",
                pos=(x, y, trunk_h / 2),
                size=(trunk_r, trunk_h / 2),
                material="bark_mat",
                contype=0,
                conaffinity=0,
            )
            # Canopy
            root.worldbody.add(
                "geom",
                name=f"tree_canopy_{i}",
                type="ellipsoid",
                pos=(x, y, trunk_h + canopy_r * 0.6),
                size=(canopy_r, canopy_r, canopy_r * 0.7),
                material="leaf_mat",
                contype=0,
                conaffinity=0,
            )

        # --- Grass blades (small tilted boxes near the fly) ---
        for i in range(n_grass_patches):
            gx = rng.uniform(-20, 20)
            gy = rng.uniform(-20, 20)
            blade_h = rng.uniform(0.8, 2.5)
            blade_tilt = rng.uniform(-0.3, 0.3)
            # Represent as thin box
            root.worldbody.add(
                "geom",
                name=f"grass_{i}",
                type="box",
                pos=(gx, gy, blade_h / 2),
                size=(0.05, 0.02, blade_h / 2),
                euler=(blade_tilt, rng.uniform(-0.2, 0.2), rng.uniform(0, math.pi)),
                material="grass_blade_mat",
                contype=0,
                conaffinity=0,
            )

        # --- Food droplets (small glowing spheres) ---
        self.food_positions = []
        for i in range(n_food_drops):
            fx = rng.uniform(-8, 8)
            fy = rng.uniform(-8, 8)
            # Avoid spawning right on top of the fly
            if abs(fx) < 2 and abs(fy) < 2:
                fx += 3.0
            self.food_positions.append((fx, fy))
            root.worldbody.add(
                "geom",
                name=f"food_{i}",
                type="sphere",
                pos=(fx, fy, food_radius),
                size=(food_radius,),
                material="food_mat",
                contype=0,
                conaffinity=0,
            )

        # --- Odor-emitting flowers (stem + petals + bright center) ---
        self.odor_positions = []
        for i in range(n_odor_sources):
            ox = rng.uniform(-12, 12)
            oy = rng.uniform(-12, 12)
            if abs(ox) < 2 and abs(oy) < 2:
                ox += 4.0
            self.odor_positions.append((ox, oy))

            stem_h = rng.uniform(1.5, 3.5)
            petal_r = rng.uniform(0.5, 1.0)

            # Stem (thin green cylinder)
            root.worldbody.add(
                "geom",
                name=f"flower_stem_{i}",
                type="cylinder",
                pos=(ox, oy, stem_h / 2),
                size=(0.06, stem_h / 2),
                material="grass_blade_mat",
                contype=0,
                conaffinity=0,
            )
            # Petals (flat ellipsoid)
            root.worldbody.add(
                "geom",
                name=f"flower_petal_{i}",
                type="ellipsoid",
                pos=(ox, oy, stem_h + 0.05),
                size=(petal_r, petal_r, petal_r * 0.2),
                material="flower_petal_mat",
                contype=0,
                conaffinity=0,
            )
            # Center (small glowing sphere)
            root.worldbody.add(
                "geom",
                name=f"flower_center_{i}",
                type="sphere",
                pos=(ox, oy, stem_h + 0.15),
                size=(petal_r * 0.3,),
                material="flower_center_mat",
                contype=0,
                conaffinity=0,
            )

        # --- Small obstacles (pebbles / debris) ---
        root.asset.add(
            "material", name="pebble_mat",
            rgba=(0.55, 0.50, 0.42, 1.0),
            reflectance=0.05,
        )
        self.obstacle_positions = []
        for i in range(n_obstacles):
            px = rng.uniform(-10, 10)
            py = rng.uniform(-10, 10)
            if abs(px) < 2 and abs(py) < 2:
                px += 3.5
            self.obstacle_positions.append((px, py))
            pebble_r = rng.uniform(0.3, 0.7)
            root.worldbody.add(
                "geom",
                name=f"obstacle_{i}",
                type="ellipsoid",
                pos=(px, py, pebble_r * 0.4),
                size=(pebble_r, pebble_r * 0.8, pebble_r * 0.4),
                euler=(0, 0, rng.uniform(0, math.pi)),
                material="pebble_mat",
                contype=0,
                conaffinity=0,
            )
