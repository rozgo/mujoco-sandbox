# Ember machinery palette

[Open the static palette preview](../../../previews/locomotion/ember/palette_v1.png)

An optional native MuJoCo theme based on the supplied Golden Ember palette.
Charcoal and steel form the neutral base; yellow painted panels provide the main
accent. Forest green balances the warm colors in small contact indicators, while
oxide red labels physical damage. These are compatible accent choices rather
than a claim that red or green is the exact complementary hue of yellow-orange.

| Color | Hex | Use |
| --- | --- | --- |
| Charcoal | `#1F1F1F` | Background, chassis and joints |
| Steel | `#4B4847` | Structure, deck and separators |
| Ivory | `#E6E1DB` | Text and exposed metal |
| Machinery yellow | `#FFC31F` | Body panels, headings and deck edge markings |
| Orange | `#E88107` | Mechanism and seam accents |
| Forest green | `#3F6B53` | Contact indicators, paired with foot labels |
| Oxide red | `#C64B3C` | Damage markers and damage captions |

Rendered material colors vary with lighting. Curved cosmetic shell panels,
damage spheres and deck-edge marks exist only in `MjvScene`; they do not add mass,
contacts, sensor surfaces or support. The trained actor does not observe these
colors. Graphite/classic defaults, vendor assets and all previous films remain
available.

## Preview and live viewers

```sh
uv tool run --from uv==0.12.12 uv run --project experiments/adaptive_locomotion --locked python experiments/adaptive_locomotion/scripts/preview_ember.py --output previews/locomotion/ember/new_preview.png
uv tool run --from uv==0.12.12 uv run --project experiments/adaptive_locomotion --locked adaptive-dog view --checkpoint assets/locomotion/checkpoints/gpu_from_scratch/unified.pt --case whole_fr --timestep 0.0005 --presentation damage --theme ember
uv tool run --from uv==0.12.12 uv run --project experiments/adaptive_locomotion --locked adaptive-balance view --checkpoint assets/locomotion/checkpoints/gpu_from_scratch/unified.pt --surface gap_fr --theme ember
uv tool run --from uv==0.12.12 uv run --project experiments/adaptive_locomotion --locked adaptive-moving view --checkpoint assets/locomotion/checkpoints/gpu_from_scratch/unified.pt --motion combined --theme ember
```

The static **1920×1200** preview was rendered and visually inspected. The native
Mac moving viewer completes a three-second launch check. Four focused tests
compare themed and original flat/moving models: physical parameters, poses,
velocities and sensors remain identical across 500 physics steps per pair.
This palette work adds **zero training** and does not alter rewards.

The synchronized three-view video layout also supports Ember. On the host with
the original accepted moving trace, replay it with the optional theme:

```sh
uv tool run --from uv==0.12.12 uv run --project experiments/adaptive_locomotion --locked python experiments/adaptive_locomotion/scripts/record_graphite.py --theme ember --output previews/locomotion/ember/moving_replay.mp4
```

This command styles the earlier accepted trace and verifies its model/trajectory
hashes. It does not show the newly trained GPU checkpoint. No replacement of the
full accepted movie was generated during this palette study.
