# Clearer damage and ground contact

User request: call the motion walking, make cut locations obvious with orange
spheres, and improve shadows so foot clearance and ground contact are easier
to see. Follow-up started 2026-09-11 00:39:35 UTC.

The `damage` presentation preset adds observer-only MuJoCo scene spheres at
actual cut locations: radius 28 mm at the remaining thigh's calf attachment,
and radius 40 mm at the original hip attachment when the whole leg is absent.
They are enlarged visual markers, not replacement feet or physical supports.
They add no mass, collisions or range-sensor geometry. The physical model and
trained policy retain their previous parameters and allowed support surfaces.

A body-following spotlight, reduced headlight fill, brighter floor and 4096-pixel
shadow map improve contact contrast. These are MuJoCo rendered shadows, not
painted ground blobs. Shadow settings follow the [official rendering guidance](https://mujoco.readthedocs.io/en/stable/XMLreference.html#visual-quality).
Cameras use matched oblique angles from the damaged side: -50 degrees for left
removals, +50 for right removals. No image mirroring is used. Status labels now
say **WALKING** while the walk is in progress.

The new video replays all nine original validated `limb_bilateral_smooth`
trajectories at the same timestamps. Checkpoint, physical-model and trajectory
hashes are checked before visual changes; render-model hashes are recorded
separately. No new RL or mission simulation is needed. Earlier videos remain
available. Orange markers indicate damage; the contact dots still use recorded
forces from permitted support surfaces.

From the repository root:

```sh
cd experiments/adaptive_locomotion
uv tool run --from uv==0.12.12 uv run --locked adaptive-dog view \
  --checkpoint ../../assets/locomotion/checkpoints/limb_bilateral_selected_seed2.pt \
  --case whole_fr --presentation damage
uv tool run --from uv==0.12.12 uv run --locked adaptive-dog grid \
  --checkpoint ../../assets/locomotion/checkpoints/limb_bilateral_selected_seed2.pt \
  --family limb_loss --seed 9143 --presentation damage \
  --replay-from ../../outputs/locomotion/recordings/limb_bilateral_smooth \
  --output ../../previews/locomotion/limb_damage_visible.mp4
```

On a fresh checkout, omit `--replay-from` to regenerate the deterministic seed
9143 demonstrations. The capture directories are generated outputs, not Git
assets. Use `--presentation original` for the prior visual settings.

[Preview](../../previews/locomotion/limb_damage_visibility_preview.png) ·
[Updated nine-case video](../../previews/locomotion/limb_damage_visible.mp4) ·
[Previous video](../../previews/locomotion/limb_bilateral_smooth.mp4)
