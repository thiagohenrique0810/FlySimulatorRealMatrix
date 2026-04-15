"""Flies in a custom environment with terrain, obstacles, and food."""

import itertools
import numpy as np
import mujoco.viewer

from flygym.compose import Fly, KinematicPosePreset, ActuatorType, FlatGroundWorld
from flygym.compose.physics import ContactParams
from flygym.anatomy import (
    Skeleton, AxisOrder, JointPreset, ActuatedDOFPreset, ContactBodiesPreset,
)
from flygym.utils.math import Rotation3D
from flygym import Simulation
from flygym_demo.spotlight_data import MotionSnippet


class ObstacleWorld(FlatGroundWorld):
    """World with terrain features, obstacles, and food items."""

    def __init__(self, name="obstacle_world", **kwargs):
        super().__init__(name=name, **kwargs)
        self._obstacle_geoms = []
        self._food_geoms = []
        self._build_environment()

    def _build_environment(self):
        root = self.mjcf_root

        # --- Materials ---
        root.asset.add(
            "texture", name="wood_tex", type="2d", builtin="flat",
            width=100, height=100, rgb1=(0.55, 0.35, 0.15), rgb2=(0.45, 0.28, 0.12),
        )
        wood_mat = root.asset.add(
            "material", name="wood", rgba=(0.55, 0.35, 0.15, 1),
            reflectance=0.05,
        )
        root.asset.add(
            "texture", name="stone_tex", type="2d", builtin="flat",
            width=100, height=100, rgb1=(0.5, 0.5, 0.5), rgb2=(0.4, 0.4, 0.4),
        )
        stone_mat = root.asset.add(
            "material", name="stone", rgba=(0.5, 0.5, 0.55, 1),
            reflectance=0.1,
        )
        food_mat = root.asset.add(
            "material", name="food", rgba=(0.9, 0.2, 0.1, 1),
            reflectance=0.3,
        )
        leaf_mat = root.asset.add(
            "material", name="leaf", rgba=(0.2, 0.6, 0.15, 0.9),
            reflectance=0.1,
        )

        # --- Ramp (inclined plane for climbing) ---
        ramp = root.worldbody.add(
            "body", name="ramp", pos=(8, 0, 0.15),
            euler=(0, -15, 0),  # tilted 15 degrees
        )
        ramp_geom = ramp.add(
            "geom", name="ramp", type="box", size=(3, 4, 0.05),
            material=wood_mat, contype=0, conaffinity=0,
        )
        self._obstacle_geoms.append(ramp_geom)

        # --- Small steps / stairs ---
        for i in range(4):
            step = root.worldbody.add(
                "body", name=f"step_{i}",
                pos=(-5 + i * 1.2, 0, 0.05 + i * 0.08),
            )
            step_geom = step.add(
                "geom", name=f"step_{i}", type="box",
                size=(0.5, 3, 0.04 + i * 0.04),
                material=stone_mat, contype=0, conaffinity=0,
            )
            self._obstacle_geoms.append(step_geom)

        # --- Cylindrical log obstacles ---
        for i, (x, y) in enumerate([(3, -2), (3, 2), (6, 0)]):
            log = root.worldbody.add(
                "body", name=f"log_{i}", pos=(x, y, 0.15),
                euler=(0, 90, 0),
            )
            log_geom = log.add(
                "geom", name=f"log_{i}", type="cylinder",
                size=(0.15, 1.5),
                material=wood_mat, contype=0, conaffinity=0,
            )
            self._obstacle_geoms.append(log_geom)

        # --- Rock-like obstacles (spheres) ---
        for i, (x, y, r) in enumerate([
            (-2, -3, 0.25), (-2, 3, 0.3), (5, -3, 0.2), (5, 3, 0.35),
        ]):
            rock = root.worldbody.add(
                "body", name=f"rock_{i}", pos=(x, y, r * 0.6),
            )
            rock_geom = rock.add(
                "geom", name=f"rock_{i}", type="sphere", size=(r,),
                material=stone_mat, contype=0, conaffinity=0,
            )
            self._obstacle_geoms.append(rock_geom)

        # --- Leaves (thin flat discs as decoration) ---
        for i, (x, y, angle) in enumerate([
            (1, -1, 30), (-3, 2, -15), (7, 1, 45), (-1, -2.5, 60),
        ]):
            leaf = root.worldbody.add(
                "body", name=f"leaf_{i}", pos=(x, y, 0.01),
                euler=(0, 0, angle),
            )
            leaf.add(
                "geom", name=f"leaf_{i}", type="cylinder",
                size=(0.4, 0.005),
                material=leaf_mat, contype=0, conaffinity=0,
            )

        # --- Food items (small red-orange spheres) ---
        food_positions = [
            (2, 0, 0.1), (-3, -1, 0.1), (7, 2, 0.1),
            (-4, 3, 0.1), (4, -2, 0.1), (0, 3, 0.1),
        ]
        for i, (x, y, z) in enumerate(food_positions):
            food_body = root.worldbody.add(
                "body", name=f"food_{i}", pos=(x, y, z),
            )
            food_geom = food_body.add(
                "geom", name=f"food_{i}", type="sphere", size=(0.12,),
                material=food_mat, contype=0, conaffinity=0,
            )
            self._food_geoms.append(food_geom)

        # --- Lighting ---
        root.worldbody.add(
            "light", name="sun", pos=(0, 0, 20), dir=(0, 0, -1),
            diffuse=(0.9, 0.9, 0.8), specular=(0.3, 0.3, 0.3),
            directional="true",
        )

    def add_obstacle_contacts(self, fly, contact_params=None):
        """Add contact pairs between a fly and all obstacle/food geoms."""
        if contact_params is None:
            contact_params = ContactParams()
        contact_bodysegs = ContactBodiesPreset.LEGS_THORAX_ABDOMEN_HEAD.to_body_segments_list()
        for obs_geom in self._obstacle_geoms + self._food_geoms:
            for seg in contact_bodysegs:
                fly_geom = fly.mjcf_root.find("geom", seg.name)
                if fly_geom is not None:
                    self.mjcf_root.contact.add(
                        "pair",
                        geom1=fly_geom,
                        geom2=obs_geom,
                        name=f"{fly.name}_{seg.name}__{obs_geom.name}",
                        friction=contact_params.get_friction_tuple(),
                        solref=contact_params.get_solref_tuple(),
                        solimp=contact_params.get_solimp_tuple(),
                        margin=contact_params.margin,
                    )


# --- Config ---
NUM_FLIES = 3
SPACING = 2.5

# --- Load experimental walking data ---
snippet = MotionSnippet()

# --- Build flies ---
axis_order = AxisOrder.YAW_PITCH_ROLL
neutral_pose = KinematicPosePreset.NEUTRAL
actuator_type = ActuatorType.POSITION
actuator_gain = 150.0

flies = []
for i in range(NUM_FLIES):
    fly = Fly(name=f"fly{i}")
    skeleton = Skeleton(axis_order=axis_order, joint_preset=JointPreset.LEGS_ONLY)
    fly.add_joints(skeleton, neutral_pose=neutral_pose)
    actuated_dofs = skeleton.get_actuated_dofs_from_preset(ActuatedDOFPreset.LEGS_ACTIVE_ONLY)
    fly.add_actuators(
        actuated_dofs, actuator_type=actuator_type,
        kp=actuator_gain, neutral_input=neutral_pose,
    )
    fly.add_leg_adhesion()
    fly.colorize()
    flies.append(fly)

flies[0].add_tracking_camera()

# --- Build world ---
world = ObstacleWorld()
for i, fly in enumerate(flies):
    y_offset = (i - (NUM_FLIES - 1) / 2) * SPACING
    world.add_fly(
        fly,
        [0, y_offset, 0.7],
        Rotation3D(format="quat", values=[1, 0, 0, 0]),
    )
    # Add contacts between fly and obstacles/food
    world.add_obstacle_contacts(fly)

# --- Inter-fly contacts ---
contact_bodysegs = ContactBodiesPreset.LEGS_THORAX_ABDOMEN_HEAD.to_body_segments_list()
contact_params = ContactParams()
for fly_a, fly_b in itertools.combinations(flies, 2):
    for seg_a in contact_bodysegs:
        for seg_b in contact_bodysegs:
            geom_a = fly_a.mjcf_root.find("geom", seg_a.name)
            geom_b = fly_b.mjcf_root.find("geom", seg_b.name)
            if geom_a is not None and geom_b is not None:
                world.mjcf_root.contact.add(
                    "pair", geom1=geom_a, geom2=geom_b,
                    name=f"{fly_a.name}_{seg_a.name}__{fly_b.name}_{seg_b.name}",
                    friction=contact_params.get_friction_tuple(),
                    solref=contact_params.get_solref_tuple(),
                    solimp=contact_params.get_solimp_tuple(),
                    margin=contact_params.margin,
                )

# --- Simulation ---
sim_timestep = 1e-4
sim = Simulation(world)

joint_angles_per_fly = {}
for fly in flies:
    joint_angles_per_fly[fly.name] = snippet.get_joint_angles(
        output_timestep=sim_timestep,
        output_dof_order=fly.get_actuated_jointdofs_order(actuator_type),
    )

sim_duration = snippet.joint_angles.shape[0] / snippet.data_fps
nsteps = int(sim_duration / sim_timestep)
phase_offsets = [0, nsteps // 3, 2 * nsteps // 3]

sim.reset()
for fly in flies:
    sim.set_leg_adhesion_states(fly.name, np.ones(6, dtype=np.bool_))

print(f"{NUM_FLIES} flies in obstacle world — close the viewer to stop.")

with mujoco.viewer.launch_passive(sim.mj_model, sim.mj_data) as viewer:
    step_idx = 0
    while viewer.is_running():
        for i, fly in enumerate(flies):
            idx = (step_idx + phase_offsets[i]) % nsteps
            target = joint_angles_per_fly[fly.name][idx, :]
            sim.set_actuator_inputs(fly.name, actuator_type, target)
        sim.step()
        viewer.sync()
        step_idx += 1
