# Scene design and physical parameters

This is the scene-review milestone. The preview is frozen with `mj_forward`; only tests advance physics. Locomotion, arm planning, contact grasping, barrier navigation, release, and task-success evaluation remain for the next milestone after scene review. No object welds, kinematic base motion, or grasp shortcuts are present.

## Coordinates and layout

SI units throughout; +Z is up, chassis +X is forward, +Y is left. Robot begins at (−2.15, −1.35, 0.78) m, facing +X, offset from the source. Source table center is (0.6, 0), destination is (0.6, 3.6); both have 1.30 × 1.40 m tops at Z = 0.84 m. The barrier is 2.4 m wide × 0.18 m thick × 1.24 m high, centered at (0, 1.8). Its feet extend to Y = 1.5 and 2.1 m. Tables and barrier are fixed world fixtures.

The robot's neutral footprint is about 2.10 × 2.36 m. The open floor extends far enough on both sides of the barrier to route this footprint with margin. Future planning must account for the full legs and carried objects, not just the chassis center. A route passing west of the barrier needs the base near X ≤ −2.5 m; this is a layout allowance, not a validated navigation trajectory.

The orange hollow mug (64 mm outer diameter, 78 mm high, plus handle) and blue block (50 × 50 × 60 mm) sit near the source's approaching edge. Four dynamic blocks provide clutter. Green squares mark destination placement regions and have no collisions. The mug uses a cylinder floor, 20 wall segments and 12 handle segments, preserving the hollow opening and handle for contact. Its body diameter and the block width fit the gripper's nominal 85 mm opening. Actual grasp robustness still needs testing.

## Masses

| Component | Mass | Basis |
| --- | ---: | --- |
| Chassis hull | 16 kg | Battery, structure, computing and drive electronics |
| Belly | 4 kg | Low central mass for stability |
| Deck | 1.2 kg | Arm mounting platform |
| Head and lenses | 0.38 kg | Sensor housing |
| Two trim strips | 0.05 kg | Visual trim with negligible load |
| Each leg | 3.02 kg | Coxa 0.75, femur 1.15, joint cover 0.12, tibia 0.85, foot 0.15 |
| Each arm pedestal | 0.45 kg | Rigid mounting bracket |
| Each Kinova Gen3 | 8.1879 kg | Upstream link inertials, unchanged |
| Each attached Robotiq | ≈0.9026 kg | Upstream inertials and mesh-derived masses; redundant coupling removed |
| Complete robot | ≈58.831 kg | MuJoCo compiled subtree mass |
| Mug | 0.21 kg | Small empty mug |
| Target block | 0.12 kg | Light rigid part |
| Each clutter block | 0.15 kg | Light rigid part |

Primitive link inertias are computed by MuJoCo from the specified geometry and mass. Menagerie inertial tensors are retained. Fixtures are immovable, so their inferred masses do not affect dynamic behavior. The floating base remains unconstrained.

## Joint and actuator limits

All driven joints use bounded actuator forces. Hinge actuator force units are N·m; position targets are radians. These are simulation design limits, not certified hardware specifications.

| Joint group | Travel (rad) | Torque cap | Position gain Kp / Kv |
| --- | --- | ---: | ---: |
| Six leg yaw joints | [−0.65, +0.65] | ±65 N·m | 400 / 35 |
| Six leg hip joints | [−0.75, +0.70] | ±120 N·m | 650 / 45 |
| Six leg knee joints | [−0.65, +0.85] | ±100 N·m | 600 / 40 |
| Gen3 J1, J3 | [−π, +π] | ±105 N·m | 2000 / 100 |
| Gen3 J2 | [−2.24, +2.24] | ±105 N·m | 2000 / 100 |
| Gen3 J4 | [−2.57, +2.57] | ±105 N·m | 2000 / 100 |
| Gen3 J5, J7 | [−π, +π] | ±52 N·m | 500 / 50 |
| Gen3 J6 | [−2.09, +2.09] | ±52 N·m | 500 / 50 |
| Each Robotiq actuator | control [0, 255] | ±5 N·m in fixed-tendon coordinates | Upstream affine actuator |

The leg caps allow margin over approximately 96 N per foot in six-foot support and 192 N in three-foot support. At roughly 0.6 m horizontal hip leverage, three-foot support can require approximately 115 N·m, close to the hip cap; dynamic gait capacity is not established by this static test. Reduced speed, support scheduling, and foot placement will be necessary during controller development.

Gen3 torque limits and gains come from the pinned Menagerie model. Additional ±π travel bounds on its continuous joints are software limits to avoid unbounded winding; arm damping is 1 N·m·s/rad with 0.02 kg·m² armature. Leg damping is 4/6/5 N·m·s/rad and armature 0.035/0.05/0.035 kg·m² for yaw/hip/knee. Arm control ranges match joint ranges exactly.

Robotiq retains its two four-bar linkages, tendon split (0.5 to each driver), three equality constraints per gripper, passive joints and contact pads. Driver travel is [0, 0.8], follower [−0.872664, +0.872664], spring link [−0.296706, +0.8], coupler [−1.57, 0] rad. The tendon torque cap is not a fingertip force limit; mechanical advantage depends on configuration. Control 0 opens, 255 closes. Left and right tendon controls are independent.

## Clearance and contact

The belly's neutral ground clearance is 0.61 m. Each leg extends outward with a 0.22 m coxa, a (0.36, 0, 0.08) m femur, and a (0.24, 0, −0.78) m tibia ending in a 40 mm radius foot. Front, middle and rear legs fan out at 45°, 90°, and 135° on each side. Arms are mounted above the leg plane on the forward deck, 0.47 m apart.

Tests sample 150 simultaneous leg configurations within yaw ±0.25, hip ±0.15, knee ±0.20 rad and check non-floor collisions. This conservative swing envelope passes; it does not establish collision-free travel across every combination of the full joint limits. Planning must still check self-collision. Only adjacent-body behavior, upstream gripper linkage exclusions and the directly attached wrist/gripper pair are excluded. All six feet, chassis, leg links, arm collision meshes, gripper pads, objects, tables and barrier participate in physical contact.

The simulation uses gravity, 2 ms steps, `implicitfast`, elliptic friction cones, 100 solver iterations and `impratio=10`. Foot friction is (1.2, 0.02, 0.002); general contact friction is (0.8, 0.01, 0.001). Robotiq contact-pad settings are retained upstream. These are starting simulation values, with grasp and gait tuning pending.

## Cameras

- `third_person`: entire scene, source, barrier and destination.
- `robot_detail`: closer view of chassis, arms and legs.
- `overhead`: layout and route-clearance inspection.
- `head`: attached to the head, initially panned 0.4 rad toward the source; 75° vertical FOV.
- `left_wrist`, `right_wrist`: upstream Gen3 cameras attached to each bracelet; 41.84° vertical FOV. They follow arm orientation, so initial views need not center the objects.

## Assets and attribution

Vendored from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) at commit `8161bba264d7fa7c99ca301e91e7fb44737676ad`. `assets/menagerie/manifest.json` contains per-file SHA-256 hashes. Vendor files are unchanged. Kinova is BSD-3-Clause; Robotiq is BSD-2-Clause, with licenses retained alongside the models.

Composition applies namespaced identifiers, actuator travel bounds, damping/armature, and the [Kinova-specific Robotiq mounting instructions](https://github.com/google-deepmind/mujoco_menagerie/blob/8161bba264d7fa7c99ca301e91e7fb44737676ad/kinova_gen3/README.md#robotiq-2f-85): remove the redundant base mount and attach the gripper base directly at `pos="0 0 -0.06149039" quat="0 -1 1 0"`. No third-party visual meshes were recreated or substituted.
