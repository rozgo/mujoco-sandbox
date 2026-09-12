# GPU curriculum replay from random weights

[Measured results and limitations](../GPU_REPORT.md) · [Exact ledger](RESULTS.json) · [Immutable recipe](CURRICULUM.json) · [Task brief](BRIEF.md)

The complete run uses one evolving actor and 4,096 CUDA physics worlds in every stage. It takes 22m 05s of learning. Final walking, static balance and moving-platform checks all use **unified.pt**. Earlier checkpoint files are ancestry milestones, not a runtime policy bank.

## Reproduce training on NVIDIA

Use a fresh checkout/output directory with the repository's locked environment. The shared reference-cache paths must not contain data from a different checkpoint; the collector checks provenance and fails rather than silently using it.

```sh
uv tool run --from uv==0.12.12 uv run --project experiments/adaptive_locomotion --locked --extra warp python experiments/adaptive_locomotion/scripts/replay_gpu_curriculum.py --recipe docs/locomotion/gpu_from_scratch/CURRICULUM.json --output outputs/locomotion/gpu_from_scratch/reproduction --through 30
```

The driver rejects a world count other than 4,096. It saves an initial state, configuration, final state and actual timing record for each stage. All references map to this new run's prior checkpoints; no historical weights are imported. It halts on incomplete stages, and does not overwrite failed attempts. See the recipe for all rewards, seeds and update counts.

On the capture host, aggregate its raw records with:

```sh
uv tool run --from uv==0.12.12 uv run --project experiments/adaptive_locomotion --locked --extra warp python experiments/adaptive_locomotion/scripts/report_gpu_curriculum.py --run outputs/locomotion/gpu_from_scratch/reproduction --recipe docs/locomotion/gpu_from_scratch/CURRICULUM.json --output outputs/locomotion/gpu_from_scratch/reproduction/ledger.json
```

## Run the final actor on macOS

Every command below loads the same checkpoint. The existing native viewer launchers handle `mjpython` on macOS.

```sh
uv tool run --from uv==0.12.12 uv run --project experiments/adaptive_locomotion --locked adaptive-dog view --checkpoint assets/locomotion/checkpoints/gpu_from_scratch/unified.pt --case whole_fr --timestep 0.0005 --presentation damage
uv tool run --from uv==0.12.12 uv run --project experiments/adaptive_locomotion --locked adaptive-balance view --checkpoint assets/locomotion/checkpoints/gpu_from_scratch/unified.pt --surface gap_fr --theme graphite
uv tool run --from uv==0.12.12 uv run --project experiments/adaptive_locomotion --locked adaptive-moving view --checkpoint assets/locomotion/checkpoints/gpu_from_scratch/unified.pt --motion combined --theme graphite
```

To repeat the frozen combined audit, choose a new output directory:

```sh
uv tool run --from uv==0.12.12 uv run --project experiments/adaptive_locomotion --locked python experiments/adaptive_locomotion/scripts/audit_shared_gpu_policy.py --checkpoint assets/locomotion/checkpoints/gpu_from_scratch/unified.pt --output outputs/locomotion/gpu_from_scratch/review --trials 4 --seed 9507
```

Evaluation is outside training time. The saved audit includes every failed trial and does not select or refine checkpoints. The 30 raw [training records](stages/) retain actual source, configuration, timing and per-rollout measurements. All intermediate raw checkpoints remain on the training host; the initial state and four major milestones are versioned with Git LFS.
