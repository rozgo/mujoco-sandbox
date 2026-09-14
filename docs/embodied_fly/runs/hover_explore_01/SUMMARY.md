# More actor updates, without meaningful improvement in hover

All three ten-second evaluation starts remain airborne and retain roughly 30 Hz
wing strokes. This PPO trial fails the declared improvement criteria and all
three original accurate-hover gates. Keep the preferred imitation actor.

| Nominal result, after the first second | Preferred imitation | This PPO trial |
| --- | ---: | ---: |
| Altitude RMS error | 8.724 mm | 8.652 mm |
| Position RMS error | 63.708 mm | 63.776 mm |
| Peak position error | 108.999 mm | 109.127 mm |
| Settled wing frequency | 29.75 Hz | 29.75 Hz |
| Mean repeated height ripple, seconds 6–10 | 0.09487 mm | 0.09489 mm |

Neither nominal improvement reaches the required 10%. Both perturbed starts stay
within the declared regression limits, but still drift. Small ripple around the
wrong mean height is not accurate hovering. All case measurements and original
quality gates remain in comparison.json and evaluation.json.

## Measured training

Training takes **396.971387 seconds (6 min 37 s)**, plus 7.726508 s setup. The
requested five minutes finishes its current update cycle, running 96.97 seconds
over that request. **524,288 action transitions across 64 hover worlds** provide
**1,048.576 seconds (17 min 28.6 s)** of aggregate simulated experience.
CPU MuJoCo/mjbatch uses 16 threads; neural computation uses the RTX 4090. The
accepted plant stays at 1,000 Hz physics and 500 Hz actor actions; this is not Warp.

Four rollouts yield 64 actor updates and 1,024 critic updates. The two initial
critic-only rollouts take 68.386636 seconds inside training time. Collection is
135.559705 seconds and optimization 261.410056 seconds. Of 64 completed training
episodes, 62 survive and two fail; failure traces are preserved. Peak PyTorch
CUDA allocation is 9,074,088,960 bytes, excluding other processes. This is continued
learning from an imitation checkpoint, not a from-scratch training claim.

The sensory encoder, sensor extension, motor decoder, wing readout and neural
cell gain/leak/bias all change under PPO. See learning_scope.json for measured
parameter-change norms. Graph routing and normalized signed adjacency stay fixed.
The utility selector stays frozen in this motor stage. No PID or imitation enters
training. The critic is separate and is not used by the deployed motor actor.

## What changed and what remains weak

Exploration std/floor grows .001 to .003 after a frozen probe: all 16 worlds
survive at .003, whereas ten fail at .006. Two failures during this training run
show that a small successful probe is not a guarantee. The actor KL guard never
triggers here; maximum KL stays below .015. More actor optimization occurred,
but it did not establish better flight.

The critic keeps its 1713 → 128 → 128 → 1 tanh MLP and adds fixed first-rollout
input statistics. It gets 16 fitting epochs per rollout instead of two; actor
epochs stay at two. Final fit RMSE is 0.801 on targets with standard deviation
0.771 and explained variance -0.0063. It still weakly distinguishes returns.
Input standardization and extra fitting have not solved value estimation. These
are bootstrap-target training metrics, not independent future-outcome accuracy.
The prior offline diagnostic's mixed and safe subsets are both preserved.

Next, make the critic learn measured outcomes reliably from recorded experience,
including checking how minibatches mix times and worlds, before another longer
flight run. Preserve the body, actor interface, preferred imitation checkpoint
and all failed trials. Accurate hover, later motor skills and survival remain open.

Eight focused tests pass; the complete package passes 184 tests in 157.62 seconds.
See PLAN.md, RUN_COMMAND.md, training.json and validation.json for reproduction.
