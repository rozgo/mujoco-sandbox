# Adaptive dog v1: frozen official baseline

The user accepted the rear-support refinement on **2026-09-11** after reviewing
`limb_rear_overlap.mp4`, describing its gait as organic and realistic, and
explicitly requested promotion to `main`. **This is the official solution and
video.** Later work should start from the `adaptive-dog-v1` Git tag on a new
branch, with new artifact names. No new training, reward changes, symmetry
fine-tuning or contact changes were made for this release.

**[Watch the official 4K video](../../previews/locomotion/adaptive_dog_v1.mp4)** ·
**[Share the looping GIF](../../previews/locomotion/adaptive_dog_v1.gif)** ·
[Machine-readable release record](OFFICIAL_V1.json)

## Exact policy and behavior

- Checkpoint: `assets/locomotion/checkpoints/adaptive_dog_v1.pt`.
- Byte-identical source: `limb_rear_overlap_strong_90s_seed2.pt`, SHA-256
  `174b3094eaf20233f151ff0ad19fc21c4612e8b1e7468e683c81ab8f980d846b`.
- One shared PPO actor: **86 → 128 → 128 → 12**, with ELU hidden activations.
  Its inputs comprise **66 observations plus 20 constant placeholders**. The
  actor has **29,196 parameters** and no recurrent network or gait clock.
- Observations: 12 joint-position offsets, 12 joint velocities, 3 body angular
  velocities, 3 up-vector components, 12 previous actions, 3 velocity commands,
  12 missing-joint bits and 9 range readings. Missing-joint information is supplied;
  autonomous damage diagnosis is not demonstrated. RGB cameras are observers.
- Output: nominal-pose joint offsets at **50 Hz**, through the existing native
  torque-limited position servos. Inactive joint slots are ignored. The original
  hip/thigh caps are **23.7 Nm**, calf caps **45.43 Nm**. MuJoCo supplies the free
  body's dynamics, gravity and physical ground contacts.
- Bodies: healthy; FL/FR/RL/RR lower-leg removal (entire calf and foot absent);
  FL/FR/RL/RR whole-leg removal. All use identical deployed weights. This is
  flat-ground single-removal locomotion, not general parkour or arbitrary damage.
- Runtime physics **0.5 ms**, control **20 ms**. Training physics was **2 ms**.
  The video plays at **1×** and follows scripted lane velocity commands.

The training progression was healthy PPO walking, physical limb-loss adaptation,
training-only healthy-motion guidance, visible-step rewards, and the final rear
support preference. The last reward discourages neither rear foot carrying load
when a front limb is damaged and both rear limbs remain intact. It leaves each
leg's timing to the policy. Recent rounds use no mirror loss; an earlier ancestor
did use one, and the later [symmetry probe](SYMMETRY_OPTION.md) remains a separate
experiment on another checkpoint.

Selected ancestry: **1922.725 seconds (32 min 03 s)** and **38,400,000 transitions**.
The last two attempted rounds cost **179.114 seconds (2 min 59 s)** and **3,624,960
transitions**, using **512 CPU MuJoCo/mjbatch environments** and Apple GPU learning.
Ancestry excludes unused later updates; failed attempts and project elapsed time
remain in [the time log](../TIME_LOG.md) and [training record](REAR_OVERLAP_TRAINING.md).

## Run or reproduce

From the repository root, after `git lfs pull`:

```sh
cd experiments/adaptive_locomotion
uv tool run --from uv==0.12.12 uv sync --locked
uv tool run --from uv==0.12.12 uv run --locked adaptive-dog demo
uv tool run --from uv==0.12.12 uv run --locked adaptive-dog demo --case lower_fr
uv tool run --from uv==0.12.12 uv run --locked adaptive-dog demo --case healthy
```

The launcher checks the v1 checkpoint hash and uses the frozen runtime settings.
On Mac it launches through `mjpython`. `demo --help` lists the nine cases;
`--seconds 5` limits a viewer smoke test. The isolated uv lockfile leaves the
other demos' dependencies untouched. A C++ compiler is needed to build mjbatch.

To record a new run of this version without overwriting the official media:

```sh
uv tool run --from uv==0.12.12 uv run --locked adaptive-dog grid \
  --checkpoint ../../assets/locomotion/checkpoints/adaptive_dog_v1.pt \
  --output ../../outputs/locomotion/v1_reproduction.mp4 \
  --seconds 12 --seed 9143 --family limb_loss --presentation damage \
  --timestep 0.0005 --label 'ADAPTIVE DOG v1 / ONE POLICY / NINE BODIES'
```

The official MP4 is a title-only rerender of the approved saved trajectories.
Its sidecar records identical checkpoint, model and trajectory hashes to the
original. The 960×540 looping GIF is an observer-only transcode at 10 fps;
neither export changes the physical motion.

## Acceptance and known limitations

User acceptance is recorded separately from automatic experiment eligibility.
The original development run completed **72/72** tasks and retained the prior
lift, healthy-gait, stride/stance, speed and body-motion gates. Both FR cases
reduced simultaneous rear swing substantially, but phase separation remained
**17.6% / 11.0%** of a cycle, below the original **25%** target. The user preferred
this natural gait; the old failed gate is preserved rather than relabeled as a
pass.

The accepted video also retains a **known numerical limitation**: lower-FR's
maximum sampled contact penetration is **8.889 mm**, exceeding the unchanged
**8 mm** target. All recorded torque caps hold and all nine video tasks use
allowed support. This release does not claim that numerical criterion passed.
See [the original video QA](REAR_OVERLAP_VIDEO_QA.json).

The previously reserved seed **20260924** is audited only after the user selected
these exact weights. Full audit reports, including any failed old gates, are
saved under `V1_*.json`; rerun with:

```sh
uv tool run --from uv==0.12.12 uv run --locked python scripts/validate_official_v1.py
```

That audit exits nonzero if the original automatic gates fail. It does not
reselect or modify the user-accepted policy. Original development reports,
previous selected policies and the original experimental-labeled video remain
available for comparison.

## Final audit and release checks

After user selection, the reserved seed **20260924** passes **288/288** full tasks
at 0.5 ms and **72/72** task checks at 0.25 ms. Every existing foot-lift,
healthy-gait, stride/stance, speed and body-motion retention gate passes. The
original quarter-cycle FR alternation gate still fails; this outcome is retained
in [the audit summary](V1_AUDIT_SUMMARY.json) and [paired comparison](V1_COMPARISON.json).

**46 package tests + 38 mjbatch tests = 84 tests passed**, Ruff passed, and the
new hash-pinned `demo` command passed a five-second native Mac viewer smoke test.
Both exports fully decode: **300 MP4 frames** and **120 GIF frames**, each covering
twelve seconds. Encoded MP4 opening/middle/end and the GIF midpoint were inspected.
Model, trajectory and checkpoint hashes match the approved original; this is
an observer-only relabel and transcode. The GIF is approximately **8.9 MB**.
