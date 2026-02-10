"""Droid X3 constants."""

from pathlib import Path

import mujoco

from mjlab import MJLAB_SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import (
  ElectricActuator,
  reflected_inertia_from_two_stage_planetary,
)
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

X3_XML: Path = (
  MJLAB_SRC_PATH / "asset_zoo" / "robots" / "x3" / "xmls" / "x3.xml"
)
assert X3_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, X3_XML.parent / "assets", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(X3_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Actuator config.
##

X3_ACTUATOR_LEGS = BuiltinPositionActuatorCfg(
  target_names_expr=(
      ".*_hip_yaw_joint",
      ".*_hip_roll_joint",
      ".*_hip_pitch_joint",
      ".*_knee_joint",
  ),
  effort_limit=100.0,
  armature=0.01,
  stiffness=200.,
  damping=3.,
)
X3_ACTUATOR_FEET = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_ankle_pitch_joint", ".*_ankle_roll_joint"),
  effort_limit=50.0,
  armature=0.01,
  stiffness=30.,
  damping=2.,
)
X3_ACTUATOR_ARMS = BuiltinPositionActuatorCfg(
  target_names_expr=(
        ".*_shoulder_pitch_joint",
        ".*_shoulder_roll_joint",
        ".*_shoulder_yaw_joint",
        ".*_elbow_joint",
        ".*_wrist_roll_joint",
        ".*_wrist_pitch_joint",
        ".*_wrist_yaw_joint",),
  effort_limit=30.0,
  armature=0.01,
  stiffness=40.,
  damping=3.,
)

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0., 0., 0.93),
  joint_pos={
    ".*_hip_pitch_joint": -0.3,
    ".*_knee_joint": 0.6,
    ".*_ankle_pitch_joint": -0.3,
    ".*_shoulder_pitch_joint": 0.0,
    ".*_elbow_joint": 0.0,
    "left_shoulder_roll_joint": 0.0,
    "right_shoulder_roll_joint": 0.0,
  },
  joint_vel={".*": 0.0},
)

KNEES_BENT_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0., 0.,0.93 ),
  joint_pos={
       ".*_hip_pitch_joint": -0.3,
            ".*_knee_joint": 0.6,
            ".*_ankle_pitch_joint": -0.3,
            ".*_elbow_joint": 0.0,
            "left_shoulder_roll_joint": 0.0,
            "left_shoulder_pitch_joint": 0.0,
            "right_shoulder_roll_joint": -0.0,
            "right_shoulder_pitch_joint": 0.0,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

# This enables all collisions, including self collisions.
# Self-collisions are given condim=1 while foot collisions
# are given condim=3.
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={r"^(left|right)_foot[1-6]_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot[1-6]_collision$": 1},
  friction={r"^(left|right)_foot[1-6]_collision$": (0.6,)},
)

FULL_COLLISION_WITHOUT_SELF = CollisionCfg(
  geom_names_expr=(".*_collision",),
  contype=0,
  conaffinity=1,
  condim={r"^(left|right)_foot[1-6]_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot[1-6]_collision$": 1},
  friction={r"^(left|right)_foot[1-6]_collision$": (0.6,)},
)

# This disables all collisions except the feet.
# Feet get condim=3, all other geoms are disabled.
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(r"^(left|right)_foot[1-6]_collision$",),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
)

##
# Final config.
##

X3_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
      X3_ACTUATOR_LEGS,
      X3_ACTUATOR_FEET,
      X3_ACTUATOR_ARMS,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_x3_robot_cfg() -> EntityCfg:
  """Get a fresh G1 robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.
  """
  return EntityCfg(
    init_state=KNEES_BENT_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=X3_ARTICULATION,
  )


X3_ACTION_SCALE: dict[str, float] = {}
for a in X3_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    X3_ACTION_SCALE[n] = 0.25 * e / s


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_x3_robot_cfg)

  viewer.launch(robot.spec.compile())
