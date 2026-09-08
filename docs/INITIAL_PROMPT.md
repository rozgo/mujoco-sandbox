# Original project brief

Build a physically simulated robot demo in MuJoCo: a six-legged, spider-like robot with two independent arms that transfers objects between tables.

Build the scene first and show me a preview before adding movement.

The robot:
- Design a clean, solid-looking chassis with six walking legs. Primitives are fine.
- Use existing Kinova Gen3 arm and Robotiq 2F-85 gripper assets from MuJoCo Menagerie.
- Give the legs enough clearance to move freely

The task:
- Approach a cluttered source table from an offset starting position.
- Pick up a small mug with one gripper and a block with the other.
- Carry both objects around a barrier to a second table.
- Place and release both objects.

Choose and document sensible masses, joint limits and actuator torque limits.

Provide head and wrist cameras, plus a third-person view.

Test as you go and fix what breaks.

---

Turn this into a git repo, gitignore whats needed, git lfs large files, use uv for python and do your best to make this work on mac...

## Follow-up

also save this initial prompt somewhere as and MD file
