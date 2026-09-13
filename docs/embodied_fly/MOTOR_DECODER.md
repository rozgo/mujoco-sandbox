# Calibrating the existing motor decoder

The deployed actor is unchanged in shape: sensory encoder → measured recurrent
graph → 815 motor-cell states → LayerNorm → 256 tanh units → 78 tanh actuator
outputs. This stage trains the existing **230,572 motor-decoder parameters**.
It adds no network, sensor bypass or runtime teacher. The upstream actor stays
bitwise fixed during supervised calibration. Physical-outcome PPO is a later,
separate stage that can update the actor's trainable cell/interface parameters.

`motor_features` replays actual recorded observations causally through a frozen
actor. Each episode has independent memory; complete variable-length histories,
including stop/resume, are retained. Padding is not recorded. The cache contains
motor-cell states, parent outputs, captured labels, episode IDs and frame IDs.
The decoder receives only motor-cell states. Parent outputs support retention;
external labels supervise wings. Whole episodes are excluded from training,
normalization uses training targets only, and hashes bind the data and checkpoint.

The initial three corpora contain 12,000 ground frames, 12,000 original flight
frames and 6,400 phase-diverse startup frames. Their split leaves **21,800
distinct training frames**. The timed fit samples ground with probability 0.5;
otherwise it chooses a flight corpus uniformly. On flight batches, 25% of samples
come from the first 25 frames. Ground loss preserves all parent outputs. Flight
loss follows wing labels while retaining the parent's 72 other outputs.

These are cached optimization examples, not newly simulated experience. Changing
shared hidden decoder weights can affect legs, even with retention. Evaluate all
fixed ground commands and uninterrupted transitions as well as unassisted flight.
The first [decoder fit](runs/motor_decoder_01/SUMMARY.md) and its
[physical-reward continuation](runs/motor_flight_ppo_probe_02/SUMMARY.md) both fail
flight and remain diagnostics. Lower imitation error does not establish control.

Example commands use new output names to preserve prior evidence:

```sh
uv run --project experiments/embodied_fly --locked python -m embodied_fly.motor_features \
  --checkpoint assets/embodied_fly/diagnostics/motor_wing_readout_01.pt \
  --graph outputs/fly_survival/malecns --role ground --worlds 64 \
  --data outputs/embodied_fly/retention_online01_01 \
  --output outputs/embodied_fly/new_motor_ground
```

Repeat with `--role flight` for the original and startup flight corpora, preserving
the same parent. Then pass all three cache directories:

```sh
uv run --project experiments/embodied_fly --locked python -m embodied_fly.motor_calibration \
  --resume assets/embodied_fly/diagnostics/motor_wing_readout_01.pt \
  --cache outputs/embodied_fly/new_motor_ground \
  --cache outputs/embodied_fly/new_motor_flight \
  --cache outputs/embodied_fly/new_motor_startup \
  --seconds 60 --lr 0.0001 --seed 58001 \
  --output outputs/embodied_fly/new_motor_fit
```

CUDA is the default for these pilots; both CLIs accept `--device cpu`. Feature
replay requires the external measured graph; decoder fitting uses verified caches
and does not load or replace the graph. A subsequent deployed actor still runs
the complete graph at every decision.
