# Wing position actuator pilot

Latest [motor review](runs/position_motion_01/SUMMARY.md): standing passes;
walking advances, and flight remains airborne, but yaw/altitude accuracy is still
incomplete. All candidates and failed gates are retained.

The full fly now has an opt-in `wing_position` physical profile. All motor
commands in this experiment use it. Existing `wing_motion` recordings and
checkpoints remain reproducible and retain their exact fingerprints.

![Same full fly at rest](../../previews/embodied_fly/wing_position_v1_static.png)

The six brain outputs specify wing angles; MuJoCo's position actuators supply
bounded torque. The custom flight model still reads the resulting physical wing
angles/speeds. The wings have zero mass/spatial inertia, collisions and aerodynamic
forces. Only their independent angular armature remains. There is no prescribed
stroke or body-position servo in the deployed controller.

[Design and predeclared pilot](runs/position_feedback_01/PLAN.md).

```sh
# Run on the GPU host using its verified graph path and fresh output directories.
uv run --project experiments/embodied_fly --locked python -m embodied_fly.position_migrate \
  --resume assets/embodied_fly/diagnostics/state_hover_retention_02.pt \
  --graph outputs/fly_survival/malecns \
  --output outputs/embodied_fly/position_initial_01 --seed 95002

uv run --project experiments/embodied_fly --locked python -m embodied_fly.motor_focus train \
  --preset wing_position --resume outputs/embodied_fly/position_initial_01/actor.pt \
  --graph outputs/fly_survival/malecns --teacher assets/embodied_fly/teachers/walking.npz \
  --output outputs/embodied_fly/position_feedback_01 --seconds 180 \
  --worlds 32 --threads 16 --sequence 32 --episode-seconds 2 --seed 95003 \
  --trainable-subset wing-feedback --lr .0003 --feedback-lr .003 \
  --teacher-mix 0 --hover-teacher-mix .8 --retain-ground \
  --ground-retention-weight 4 --nonwing-retention-weight 4 \
  --ground-wing-loss 10 --ground-posture --hover-reference state

uv run --project experiments/embodied_fly --locked python -m embodied_fly.motor_focus evaluate \
  --preset wing_position --resume outputs/embodied_fly/position_feedback_01/actor.pt \
  --graph outputs/fly_survival/malecns \
  --output outputs/embodied_fly/position_feedback_01_evaluation \
  --seconds 5 --seed 95013 --neural-view
```

Use the native rendering backend on macOS; headless Linux can set `MUJOCO_GL=egl`.
An initialization or an assisted reference is not a trained autonomous result.
Each checkpoint must pass complete, teacher-free evaluation before promotion.

The full-body follow-up resumes `position_feedback_01.pt` with
`--trainable-subset all --lr .0001 --stand-initial-form --hover-teacher-mix 1
--nonwing-retention-weight 0`, seed 95023. Its unassisted review uses seed 95033.
Other world counts, physical clocks and loss weights remain as above. See the
[predeclared full-body plan](runs/position_fullbody_01/PLAN.md).

The moving-reference pilot resumes `position_fullbody_01.pt`, removes
`--retain-ground`, selects `--walking-reference anchored --walk-teacher-mix 1
--hover-teacher-mix 0 --hover-start-weight 10`, and uses seed 96003. Other
world counts, clocks and motor loss weights remain unchanged. Its complete
unassisted review uses seed 96013. See the
[predeclared plan](runs/position_motion_01/PLAN.md).
