# Learned thread insertion: training and results

The policy learns the approach and the moment to start the stroke. It steers stage X, Y and Z
velocities at 50 Hz from 24 observations and has a fourth action that starts the stroke. The stroke,
the lift and the thread reload are the approved cycle's programmed motions (`tube_cycle.py`).
Training uses the C environment (`native/insert_core.h`, CPU MuJoCo, 1 ms physics) with PufferLib 5.0
at the pinned revision. The network trained on the RTX 4090.

Each episode draws one random site on a phantom turned by a random angle (the vessels turn with it).
It places 0-5 threads at earlier sites. It draws the thread end's offset and release jump from 60
full-simulation runs (`THREAD_END_STATS.json`) and a disturbance level uniform in 0-2 (robot, sensing,
tissue). Rewards:

- Potential shaping toward the hover pose that puts the thread end over the target.
- Penalties for jerky approach motion and for passing low over vessels.
- −2 for robot–phantom contact, touching a placed thread, dragging the waiting thread into tissue,
  missing the thread end, or a nonfinite state.
- 2·exp(−placement / 10 µm) on release.

## Runs

| Run | Environment | Steps | Wall | Outcome |
|---|---|---|---|---|
| insert_v1 | stages could still move after the stroke started | 65.5M | 20 min | stopped: that freedom is unphysical (the needle would move sideways in tissue) |
| insert_v2 | stages held from the start of the stroke; stroke timed as the full cycle (40 ms down, 5 ms, 25 ms insert, 30 ms at depth, release, 10 ms, 30 ms back) | 299.9M | 1 h 39 min | completed |

insert_v2 used 2048 environments on 28 CPU threads, a 2-layer, 128-unit recurrent network, learning
rate 0.001 and gamma 0.995. It ran at 46-55K steps/s from source commit f2148cc. The checkpoint is
`assets/neural_insertion/insert_v2_policy.bin` (sha256 `e400ffca…616b`) and the logged config is
`insert_v2_run.ini`.

## Results (final checkpoint)

**C environment** (`policy/insert_v2_300M.json`): 200 evaluation seeds (2,000,000-2,000,199) per
level. Each cell gives the share within 10 µm, then the median placement error.

| Level | Learned, deterministic | Learned, sampled | Compensating yardstick | Aiming yardstick |
|---|---|---|---|---|
| 0 | 82.0%, 5.9 µm | 83.5%, 4.7 µm | 83.5%, 5.2 µm | 2.5%, 19.7 µm |
| 1 | 84.0%, 5.5 µm | 81.0%, 5.9 µm | 80.5%, 5.6 µm | 3.0%, 20.0 µm |
| 2 | 69.0%, 7.6 µm | 60.0%, 8.5 µm | 64.5%, 7.6 µm | 6.0%, 20.4 µm |

The learned policy touched a placed thread in 0-1% of episodes; both yardsticks did so in 2.5%.

**Full simulation** (`policy_cycle.py`, the DER thread in its tube; `policy/insert_v2_300M*.json`):
three sites per run on the approved baseline's seeds (1001-1010 at levels 1 and 2). Each cell gives
threads within 10 µm out of 30, then the median.

| Level | Learned, deterministic | Learned, sampled | Compensating yardstick | Approved cycle (`TUBE_BASELINE.json`) |
|---|---|---|---|---|
| 1 | 20/30, 8.0 µm | 19/30, 6.6 µm | 19/30, 7.7 µm | 11/30, 23 µm |
| 2 | 12/30, 11.6 µm | 15/30, 10.1 µm | 18/30, 6.8 µm | 10/30, 25 µm |

In all four columns, 60 of 60 threads were placed and every run completed. The learned and
compensating runs had no thread touches or phantom contacts; the approved cycle's baseline did not
measure thread touches.

With 30 sites per level, differences of about 5 are within run-to-run noise. In the C environment the
learned policy matches or beats the compensating yardstick. In the full simulation it matches it at
level 1 and trails it at level 2. That gap between the simulations is not yet explained.

**Where the error comes from** (full simulation, deterministic): when the stroke starts, the thread
end is 6.0 µm from the target (median; 11.8 µm at the 90th percentile). After the programmed stroke and
release it ends 9.7 µm away (22.1 µm at the 90th percentile).

## Findings and open issues

- **Trigger.** insert_v1's deterministic mean never started the stroke; the learned noise did, with a
  state-dependent probability. insert_v2's mean starts it. Evaluations report both.
- **Thread jitter at depth (full simulation).** The needle shaft's contact with the thread it carries
  fights the needle bond. The gripped segments then jitter at up to 515 rad/s while held at depth, in
  the yardstick runs too. A test with that contact off cut it about 8×. It likely inflates release
  jumps, and it caused one divergence (insert_v2 at 65.5M, level 1, seed 1009). The fix is proposed,
  not applied.
- The full-simulation sites 3 and 4 lie slightly outside the training site range. The three-site
  cycle uses sites 0, 5 and 1, which are inside it.

## Reproduce

```sh
uv run --locked python -m sixlegs.neural_insertion.insert_env --output docs/neural_insertion/INSERT_BASELINE_C_V2.json
uv run --locked python scripts/evaluate_insert_policy.py NAME --weights assets/neural_insertion/insert_v2_policy.bin [--deterministic]
uv run --locked python -m sixlegs.neural_insertion.policy_cycle --weights assets/neural_insertion/insert_v2_policy.bin --deterministic --level 1 --seed 1001 --output DIR
```

Videos are in `previews/neural_insertion/policy/` (replays of recorded full-simulation states).
