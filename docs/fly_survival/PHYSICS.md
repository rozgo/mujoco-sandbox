# Physical specification

The isolated FlyGym 2.1.0 stack uses **millimeters, grams, seconds** throughout,
including its inherited gravity of 9,810 mm/s². This deliberately preserves the
model's validated unit convention. Multiply mass by 10⁻³, force by 10⁻⁶ and torque
by 10⁻⁹ to report kg, N and Nm. World +z points upward; a fly's local +x is forward.

| Component | Choice | Source / rationale |
|---|---|---|
| NeuroMechFly mass | 1.02431 mg per fly | Sum of original FlyGym anatomical bodies |
| Leg position actuators | 42, ±65 nNm each | Existing FlyGym hybrid walking recipe |
| Foot adhesion | Six, 40 µN maximum commanded force each | Existing FlyGym attachment model |
| Leg position gain | 45 model units | Existing controller recipe |
| Joint stiffness / damping | 0.05 / 0.06 model units | Existing controller recipe |
| Passive tarsal stiffness / damping | 7.5 / 0.01 model units | Existing controller recipe |
| Fly joint ranges | Original per-joint anatomical limits | Read from the compiled model; no blanket enlargement |
| Swatter arm / pad mass | 15 / 25 mg | Illustrative lightweight laboratory actuator |
| Swatter hinge | −1.25 to +0.035 rad | Raised clear through floor-directed stroke |
| Swatter torque limit | ±12 µNm | Bounded; compare measured impact with survival model |
| Swatter position gain / velocity damping | 20,000 / 250 model units | Explicit finite position servo |
| Swatter hinge damping / armature | 80 / 0.1 model units | Damped actuator, no kinematic target tracking |
| Arena | 52 × 36 mm | Eight interacting flies, finite resources and refuges |
| Shelter clearance | Roof underside 3 mm above floor | Supports remain physical and visible |
| Terrain friction | Sliding 1.0; torsional 0.005; rolling 0.0001 | Illustrative nonslip laboratory surface |
| Physics | MuJoCo 3.9, 0.1 ms, Euler/Newton | FlyGym 2.1 supported stack; no-slip iterations 0 |
| Walking control | 1 ms, ten physics steps per command | Existing hybrid skill, sampled at 1 kHz |

Visual paint, ribs and floor seams have zero mass and zero collision masks.
They never contribute hidden support. The floor slab is recessed 0.02 mm beneath
the collision plane to avoid coplanar rendering artifacts. Swatter pad, arm,
rails, resource plinths, shelter roofs and pillars collide with flies. Each fly
has its own collision bit, permitting inter-fly collisions while excluding its
own links; original explicit ground contact pairs remain. No fly base is driven
kinematically.

Heat, damage, needs and consumption are illustrative biological abstractions,
separate from MuJoCo's rigid-body/contact and rendering systems. Their parameters
and validation will be recorded alongside the survival implementation.
