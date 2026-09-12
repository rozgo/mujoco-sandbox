# Learning journal: from a utility selector to an embodied neural controller

This is the permanent account for future demos, videos and technical reporting.
Keep failed hypotheses as well as successful runs. The current project is in
progress; its complete flight/survival curriculum has not passed acceptance.

Project began **2026-09-12 18:05:28 UTC**. The approved architecture correction
began **19:35:03 UTC**. Clocked milestones are in [TIME_LOG](../TIME_LOG.md).
Development elapsed time, setup, data generation, optimization, physical evaluation
and rendering are different quantities and must be reported separately.

## What changed, and what we learned

| Stage | What we tried | What the evidence showed | Decision |
| --- | --- | --- | --- |
| Original survival lab | Eight physical FlyGym flies, needs, food/water, shelters, swatter, heat model, eyes and full frozen MaleCNS dynamics | A working physical/inspection scaffold; body movement came from FlyGym's supplied controller | Preserve arena/UI, but do not call this learned locomotion |
| Utility CEM 01 | Optimize 72 utility weights with eight CPU worlds | 80 episodes, 302.285 s optimization, zero numerical failures; ingestion behavior needed correction | Retain diagnostic; add local taste/ingestion and a stable intake hold |
| Utility CEM 02 | Repeat search after ingestion fixes | 96 episodes, 320.892 s optimization; all four held-out flies survived, but behavior camped at water and consumed no food | Survival alone was an insufficient success criterion |
| Neural ablation | Remove neural features from the CEM 02 evaluation | Mean score was essentially unchanged, 21.584 with features versus about 21.545 without | No demonstrated useful causal contribution from the frozen connectome |
| Utility CEM 03 | Balance needs and suppress irrelevant activity scores | Interrupted during the research pause; completed generations remain diagnostic | Do not turn an unfinished run into a result |
| Course correction | Train body actuation and utility inside one MaleCNS-based recurrent actor | User approved staged locomotion, recovery, wings, needs and social behavior | Replace the external gait dependency at student evaluation |
| Complete body | Preserve legs, wings, mouth and antennae in one physical model | FlyBody compiles with 108 DoF and 78 bounded actuators, free root, approximately 0.985 mg mass | Keep full action/observation schema across curriculum stages |
| Graph computation | Fixed measured adjacency with trainable cell dynamics and learned input/utility/motor interfaces | Dense-reference gradient checks, internal parameter changes, independent memory and a no-bypass test pass | Proceed to the full graph and GPU optimization |
| Teacher bridge | Convert the published learned walking policy to modern PyTorch inference | Numerical reference samples pass; physical complete-body holds and three speeds work in short tests | Use inherited teacher knowledge for supervised initialization only |
| Tight-turn demonstrations 01 | Hold/walk plus ±1.5 rad/s turns at all speeds | Three of sixteen two-second cases toppled: episodes 8, 9 and 12; zero numerical warnings | Separate physical failure from solver failure; retain rejected demonstrations |
| Demonstrations 02 | Holds/straight walking plus broad ±0.75 rad/s turns at ≥1 cm/s | Sixteen two-second Mac cases remained upright; GPU-workstation CPU collection repeated the recipe | Start imitation with this bounded motor curriculum |
| Full graph, eight sequences | Twenty-second CUDA optimization probe | Held-out motor MSE 0.65634 → 0.06988; internal core gradients and changes recorded | Full measured graph can be trained within a small pilot budget |
| Full graph, 32 sequences | Same short probe with larger neural batch | 32,512 supervised examples in 20.098 s; about 2.93 GB peak CUDA allocation | Use 32 sequences for the five-minute motor pilot |
| Motor BC 01 | Five-minute recurrent imitation, followed by teacher-free physical tests | Held-out MSE fell to 0.01630; **0/6 physical cases passed**, despite zero solver warnings | Retain failure; diagnose feedback/state-distribution mismatch before spending more training time |
| Passive-channel diagnostic | Same BC 01 checkpoint; set the 19 wing/mouth/antenna channels to the teacher's passive command, leaving 59 walking channels learned | All three paired hold/slow/walk cases became upright and had no disallowed ground support; **0/3 still passed command tracking** | Inactive appendage commands mattered; preserve this explicit walking curriculum while improving starts and feedback control |

Original CEM reports and checkpoints remain under
`docs/fly_survival/training/`, `docs/fly_survival/evaluation/` and
`assets/fly_survival/diagnostic_cem_*.npz`. Their source commits, seeds and hashes
are in those reports. The original gait controller and the tiny utility learner
are not the architecture of the new embodied actor.

## Exactly what the new actor learns

Current motor-stage schema:

```
383 causal body/command/need inputs
  → learned 383 → 128 → 11,238 sensory encoder
  → persistent state on 166,700 MaleCNS neurons
      25,582,938 fixed, signed, normalized connections
      learned excitability, leak and bias: 500,100 internal parameters
  → 1,314 descending states → 128 → six utility scores
  → selected intention encoded back into the same recurrent core
  → 815 motor-neuron states → 256 → 78 bounded actuator outputs
```

Layer normalization is used in the utility/motor readouts. There are **2,409,132
trainable parameters** in total. Four internal neural updates occur per motor
decision. State is a signed latent/rate-style computation with assumed dynamics,
not biologically calibrated voltage or measured spikes. The topology and neuron
identities come from the measured dataset. We are **not training 25.6 million
individual edge weights** in this version; the 500,100 trainable internal cell
parameters change computation inside that fixed topology.

Utilities remain an actual decision head. A winner feeds a learned intention
projection back into the graph before motor decoding. It does not call a walking
function. All actor parameters are saved together; each fly has independent memory.
The first warm start labels rest/explore from the commanded motor task. Feeding,
drinking, escape and grooming output slots exist, but are not acquired skills yet.

The student sees current joint state, actuator activation, local velocity and
orientation, foot contact, previous action, command and needs. Eye cameras are
rendered for observation at this stage; pixels are not student inputs yet. The
teacher alone receives the desired future COM path. Student evaluation imports
no teacher and supplies no gait phase or future trajectory.

## Measured compute: quantities that must stay distinct

| Run | Optimization | Parallel neural sequences | Supervised examples presented | Unique training frames | Initial → final held-out motor MSE | Peak allocated CUDA memory |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| Graph probe 01 | 20.023827 s | 8 | 9,728 | 12,000 | 0.656341 → 0.069877 | 1,295,849,472 bytes |
| Graph probe 32 | 20.097561 s | 32 | 32,512 | 12,000 | 0.649047 → 0.061284 | 2,930,399,232 bytes |
| Motor BC 01 | 300.033296 s | 32 | 484,864 | 12,000 | 0.649047 → 0.016298 | 2,930,399,232 bytes |

These probes use an **RTX 4090** for forward/backward passes. Eight and 32 refer
to independent demonstration sequences, **not live physical worlds**. Each update
uses eight context steps without gradient, then eight supervised recurrent steps.
The 12,000 unique training frames represent 24 seconds of recorded body experience;
repeated presentations do not create new physical experience. Validation uses
separate complete episodes. Its sampled batch differs between the two probes,
so the table is not a controlled comparison of final motor quality.

Training throughput was approximately 486 versus 1,618 supervised examples/s
including optimization. Full-graph setup took 3.884 s and 2.814 s respectively;
post-training validation took 0.085 s and 0.099 s. They are outside optimization
time. Whole-scene dynamics use **native CPU MuJoCo**, at 5 kHz physics and 500 Hz
motor decisions. MuJoCo Warp is an optional dependency and has not powered these
motor-learning runs. The earlier Warp fly hold was a separate physics-only probe.

Motor BC 01 performed **1,894 optimizer updates**. Setup took 2.764 s and its
offline post-training validation 0.099 s, outside the measured five-minute run.
The held-out imitation error reduction is about 97.5%, but **must not be presented
as a 97.5% improvement in walking**. In six two-second free-body evaluations,
five failed the tilt/height stability test; the fast-walk case stayed within that
stability envelope but failed speed, turning and allowed-support requirements.
All six failed overall, with no numerical failures. This is direct evidence that
offline action imitation is insufficient so far. The shared checkpoint is
`7bf16a8033a1013088e9672acbd9d44e4c08194c4ff8ddd744c270e9ae874243`.
Exact reports and curves are in [`runs/`](runs/); three diagnostic checkpoints are
retained under `assets/embodied_fly/diagnostics/` through Git LFS.

On the Mac, the four short teacher probes collected two physical seconds in
1.600 s after 2.922 s setup. Demonstration collection 01 collected 32 physical
seconds in 25.925 s after 2.929 s setup; collection 02 took 25.384 s after 2.918 s
setup. These are teacher/data costs, not training our graph. The GPU workstation
collected the 32-second dataset in **42.883 s after 6.142 s setup**, using CPU
physics on that machine. Do not substitute Mac numbers for its timing.

A subsequent replay diagnostic fed the frozen student a complete two-second
teacher observation sequence. MSE was 0.02009 overall, 0.06767 for the first 32
frames and 0.01703 for the last 500 frames. Neural-state standard deviation changed
from 0.1155 to 0.2087; this case did not show runaway recurrent-state error.
Predicted wing commands had normalized RMS 0.0323 despite a zero-wing-command
teacher target. This motivates testing passive wing/mouth/antenna channels during
the initial walking curriculum and training cold starts/feedback corrections.
The subsequent paired passive-channel diagnostic supported that concern: without
additional training, all three tested cases stayed upright (minimum vertical-axis
alignment ≥0.987) and had zero disallowed ground support. They still moved/turned
incorrectly for their commands, so none passed overall. The intervention muted
wings, mouth and antennae together; it does not isolate wing commands alone.
This is a declared first-stage actuator mask, not a new gait or a second policy.
All anatomy and degrees of freedom remain physical, and the actor still has 78
output slots for later curriculum stages. The mask must not be hidden in a claim
that flight or complete appendage control has already been learned.

The raw GPU captures remain in ignored run directories for re-rendering. Published
diagnostic reports/checkpoints are indexed with SHA-256 in
[`EXPERIMENT_INDEX.json`](EXPERIMENT_INDEX.json). This index also links the prior
utility experiments, so the narrative includes both architectures.

## Compatibility work and provenance

- FlyBody source is pinned at `d015e9bfe441bd90ae431bac24c55cb74bdbce26`.
  Preserve upstream physical parameters in CGS and convert reported metrics to SI.
- Python 3.14 installation encountered labmaze's missing wheel/Bazel build
  requirement. Excluding that dependency then failed dm-control arena import.
  Use compatible Python 3.12 with current MuJoCo 3.13.0, PyTorch 2.14.0 and NumPy
  2.5.3; keep the failed compatibility route documented rather than silently patched.
- Official teacher export uses isolated Python 3.13, TensorFlow 2.21.0 and current
  available TensorFlow Probability 0.25.0. Its old saved distribution type names
  need two explicit aliases during export. Exported weights include numerical
  reference inputs/outputs; runtime checks float32 agreement before use.
- Teacher weights came from the official Figshare data, whose record says
  GPL-3.0-or-later. Their original link, authors, checksum, conversion description
  and license are in `assets/embodied_fly/teachers/`. The body-code license is
  Apache-2.0. Do not attribute the teacher's original training time to our runs.
- Implementation was performed agentically through repository inspection,
  source/primary-paper research, coding, native tests, SSH/Git GPU execution and
  physical capture. Record the actual runtime/model identity if available;
  do not infer a specific model version from a draft social post.

## Reporting rules for films and demos

1. Show the source commit, checkpoint hash, data recipe and evaluation report
   behind each selected result. Save unsuccessful cases too.
2. Say **supervised imitation warm start** for these graph probes. They are not
   PPO or a demonstration of reward-driven survival learning. Later RL gets its
   own timed stage, while extending the same actor.
3. Falling, bad foot support and numerical failure are separate outcomes. Physical
   evaluation checks tilt, height, speed, turning and disallowed ground loads.
4. “Internal neural parameters learned” has gradient/weight evidence. “Measured
   topology improved learning” still requires a matched frozen/rewired control.
5. A successful teacher clip is labeled **reference teacher**. A low imitation
   error is not a successful free-body rollout, learned flight or reconstructed fly.
6. State whether a film is captured simulation, replay or generative styling, and
   identify observer-only camera feeds. Keep 1× unless another multiplier is shown.

The intended story is the learning process: build an embodied system, test a
hypothesis, retain its failure, change the architecture for a concrete reason,
then measure what the new controller actually learns.

### Feedback-data continuation plan (2026-09-12)

BC 01 remains the parent, with identical graph and 2,409,132-parameter actor.
The next short pilot introduces two explicit data/training changes: collect
teacher labels on states reached by a 25% student / 75% teacher actuator mixture,
and supervise the first eight reset frames in 25% of optimizer batches. Previously
every loss followed eight burn-in frames, leaving those initial decisions out.
The old demonstrations remain in the training pool, with whole-episode validation
splits preserved separately for each dataset. Observation normalization is retained
from the parent. BC 01 did not save Adam state, so the first continuation must
restart Adam; new checkpoints include optimizer state for subsequent resumes.

All 108 physical degrees of freedom remain. The declared walking curriculum holds
19 wing/mouth/antenna commands at raw zero; 59 channels come from the learned actor.
This is the exact intervention tested in the paired diagnostic, now shared by
collection and evaluation. No teacher may be used in student evaluation. Dataset
files distinguish teacher target actions from the executed physical mixture and
retain the parent hash, seed, collection time and failures. This is supervised
dataset aggregation, not PPO. The teacher still has a future path that the student
does not observe, so off-path target ambiguity remains a limitation to investigate
if this feedback-data pilot does not transfer.

### Tracking-frame correction before continuation

A check against the inherited teacher's actual displacement exposed a metric bug:
`mj_objectVelocity(..., mjOBJ_BODY, ..., local=1)` returns velocity in the body's
principal-inertia axes. The fly's inertia frame is rotated relative to its thorax.
Forward-speed and yaw metrics must use `mjOBJ_XBODY` (anatomical body axes).
The old BC01 reports are retained, but their speed/yaw values cannot establish
command accuracy. Tilt, height, support loads and numerical status are unaffected.
The initial three-case passive-mask result therefore proves stable support, with
command success requiring corrected evaluation. Existing trained velocity inputs
retain their original inertia coordinates; all six components remain available,
so no input reinterpretation or weight change is hidden in this evaluator fix.
A regression test applies known forward and yaw velocities at a rotated heading.
This distinction is explicit in the
[MuJoCo engine](https://github.com/google-deepmind/mujoco/blob/main/src/engine/engine_core_util.c).

### Preserved interrupted utility search

CEM 03 was interrupted for the approved full-body architecture change. Its three
completed generations and intermediate checkpoints are now retained in
`docs/fly_survival/training/cem_utility_03_interrupted/` and the experiment index.
The last complete generation reports 231.865 seconds of training. Final total
time is unknown; partial work is not reconstructed or counted as a completed run.

### Feedback continuation 01: measured outcome

The RTX 4090 continuation completed **300.050483 s** of optimization after
**3.595310 s** setup, followed by **0.148990 s** offline validation. It performed
**2,059 updates**, including **523 reset-start batches**, and **527,104** supervised
frame presentations from **24,000 unique training frames (48 physical seconds)**.
Peak CUDA allocation was **2,930,873,344 bytes**. Twelve original episodes and
twelve mixture episodes trained the same actor; eight separate full episodes were
held out. Parent normalization was retained, Adam restarted because BC01 had not
saved its optimizer, and all three internal cell-parameter groups changed.
Checkpoint: `6e946635e0dbdcc1471529b573c8be4d6a7abc207b8fab7766530302cfa1c1d9`.
The selected ancestry has **600.083779 s** of optimization; separate initial
20-second probes and data/evaluation/rendering costs are additional work.

On the continuation's fixed held-out batch, motor MSE changed **0.013827 → 0.011609**;
reset-window MSE changed **0.185662 → 0.010461**. These are imitation errors, not
physical success. Both the original masked student and the continuation stayed
upright with zero prohibited support in all six two-second development cases.
The continuation's stop displacement decreased from approximately **15.9 mm to
8.5 mm**, still far too much for a stationary fly. Its right-turn behavior worsened.
No checkpoint is accepted as a complete locomotion solution.

The utility traces expose a shortcut: BC01 selected explore for every stop frame.
The continuation selected rest for **23.6%** of the stop trial, returning to explore
while moving. The next dataset therefore uses **100% student physical control**,
with teacher labels recorded but not executed, to provide rest targets on moving
states. It retains all previous data rather than replacing walking experience.
This is a further dataset-aggregation round, not a new policy or a scripted stop.

Teacher replay in anatomical axes also showed **0.82–4.84 rad/s raw yaw RMSE**,
despite accurate mean direction. Even the teacher fails the old 0.5 rad/s
instantaneous yaw threshold. The new tracking diagnostic reports both raw and
100 ms block-mean errors without replacing the recorded gates. The 1 cm/s teacher
example averages **1.0002 cm/s**, with block errors **0.0465 cm/s / 0.2377 rad/s**.
Raw jerk, sustained drift, tilt and support should remain separate quantities.
Replay metrics use captured pre-action poses/velocities; live reports sample after
the control step and can differ at substep phase. Compare checkpoints using the
same metric path and window.

Reports, learning curves, collection manifest, utility traces and the new student
film are preserved in `runs/motor_dagger_01/` and `previews/embodied_fly/`. The film
remains labeled as a motor-learning diagnostic.

### Feedback continuation 02: holding acquired, walking lost

The 100%-student collection ran **31.188 physical seconds** in **91.030721 s**
after **6.626658 s** setup. Fifteen episodes finished; episode 9 fell at 1.188 s.
Its failure and partial trajectory are retained, and that episode was excluded
from this imitation dataset. Teacher controls were recorded as labels, not
executed. A long replay of the teacher's original hold observations produced
**99.8% rest decisions** in the prior student, compared with **23.6%** during its
physical hold trial. That supports physical-state distribution shift as a problem,
without proving it is the only cause.

Continuation 02 preserved the actor, hyperparameters, old data, normalization and
Adam state. It used **300.009422 s** of optimization, **3.435756 s** setup and
**0.150242 s** offline validation, with **2,054 updates / 511 reset-start batches /
525,824 frame presentations**. There were **36,000 unique training frames (72 s)**
from 36 full episodes, with 11 other full episodes held out across the three data
pools. Peak CUDA allocation was **2,930,971,648 bytes**. Current reports bind every
episode file to SHA-256, including rejected data. Checkpoint:
`87384d048a946ac1175084538ac49a638518e06e607ec82521633104b9b7d102`.
Its ancestry contains **900.093201 s** of optimization; this is not a successful
15-minute complete-fly training claim.

Held-out action MSE changed **0.085706 → 0.051328**, but physical results regressed.
The hold case moved only **0.198 mm net** in two seconds with valid foot support.
The slow-walk command produced almost no travel. The walk, fast-walk and both
turns fell, with prohibited support. All six still failed the original aggregate
gates, and all had zero numerical warnings. The earlier walking checkpoint remains
available; do not stitch these checkpoints into a claim of one successful policy.
This continuation is retained as a failure to preserve skills.

### Receding teacher diagnostic, not adopted as a solution

A new optional `--reference-mode receding` anchors the teacher's command-derived
lookahead to current xy/heading, removing its request to return to an absolute
path invisible to the student. It leaves the physical body untouched and keeps
the intended standing height. A regression test verifies invariance to global
translation/heading and no pose/time mutation. This is training-only reference
construction, not a deployed movement controller.

Sixteen two-second Mac teacher probes took **29.665335 s** collection after
**2.991828 s** setup; all stayed upright. However, low-speed tracking and some turns
were poor: at a 0.5 cm/s command, measured mean speed ranged about **0.077–0.266
cm/s**. One 1 cm/s negative-turn case turned the wrong way. The diagnostic is saved
under `runs/receding_teacher_probe_01/`; it has not trained an accepted student.
It does not justify another blind extension of the same imitation recipe.

The next implementation stage is the already approved physical-outcome RL stage:
start from a retained walking-capable checkpoint, reward actual command tracking,
stationary hold and valid support, and preserve learned stepping through explicit
rehearsal. Keep the same deployed graph actor and learned utility head. A separate
training-only critic is allowed; no teacher, gait generator or action selector may
replace the actor at evaluation. Native/GPU throughput and recurrent rollout
requirements must be measured before selecting the live world count.

### Actual neural-activity video overlay

The optional observer capture bins all **140,638 located neurons** into a cached
128 × 128 map at 50 Hz; **26,062 cells without locations remain in computation**.
Each bin records mean signed and mean absolute recurrent latent activity. Fixed
amber/teal colors show sign; these are not physiological spikes. The unit test
checks independent worlds, sign/magnitude averaging and unchanged source state.
The overlay does not change the actor, observations or rewards.

Two separate diagnostic films show BC01 walking and DAgger02 holding with this
overlay. Each is **2 s / 100 frames / 50 fps / 1×**, fully decoded and visually
inspected. Rendering took **7.857275 s** and **8.065591 s** on the GPU machine.
Their source/checkpoint/state/model hashes are retained. An initial setup rejection
of legacy BC01's absent metadata digest is also recorded; the loader now checks
exact neuron routing and graph weights for legacy files, and additionally checks
metadata digests when stored. No simulation ran in that rejected setup.

The user reconfirmed that flight remains required. Full-body wing control, takeoff,
flight, landing, learned survival utility, multi-agent integration and the final
combined demonstration are still open. These prototypes do not complete that goal.

### Native live-physics batching for outcome learning

Added a separate `FlyBatch` using the repository's pinned, Apache-2.0 mjbatch
source (revision `77966f85bcd8f7ef4351cb4a1a6f42e133d19725`, MuJoCo 3.13 build).
It retains full anatomy, passive-channel walking curriculum, all force bounds,
500 Hz control / 5 kHz physics, and the existing 383-value observation convention.
Additional sensors expose anatomical/inertial-frame velocity, claw contacts and
forbidden support. Reward support samples every physics substep; authoritative
acceptance still enumerates every contact in the original evaluator. The training
sensor selects the maximum-force-norm contact per geom, so is explicitly a proxy
for the acceptance maximum over every normal reaction.

A 100-action regression test produced identical single/batched poses and float32
observations, matching anatomical velocities and forbidden contact loads. It also
checks independent worlds and repeated partial resets. The first implementation
lost the folded-wing initialization on repeated resets because mjbatch only
applies changed array elements. Reset-then-initialize-then-forward fixes this;
the regression includes the previously failing repeated-reset case. No training
used that defective implementation. Physics remains CPU-based; CUDA is reserved
for the full-graph actor and learning in this path.

### First physical-outcome PPO recipe (declared before the run)

Start from **motor_dagger_01**, which retained upright, permitted-foot support in
all six command cases and partially learned rest. The later DAgger02 checkpoint
lost four locomotion cases and is not this warm start. Keep the complete measured
graph, 2,409,132 actor parameters, 383 causal inputs, 78 output channels, four
internal recurrent updates per 2 ms action, and the walking-stage 19-channel
passive mask. Graph neuron dynamics and both utility/motor interfaces receive
physical-reward gradients. No teacher executes actions during PPO collection.

Use 32 independent native CPU worlds with 16 threads and CUDA neural computation.
The physics-only 1/8/16/32-world probe measured approximately 588/3,442/5,021/6,159
transitions/s, including sensor extraction; those numbers exclude neural inference
and optimization. The initial PPO rollout is 64 actions × 32 worlds, two epochs,
eight-step recurrent chunks, gamma 0.995, GAE lambda 0.95, learning rate 1e-5,
ratio clip 0.2 and KL early stopping at 0.03. Initial motor noise is 0.04 in
pre-tanh action coordinates. Utility is sampled from the same six learned scores;
it conditions the same graph before the motor readout. PPO uses the joint
categorical-plus-squashed-Gaussian probability, without a straight-through
categorical gradient. The unchanged greedy actor is used for evaluation.

A separate training-only critic consumes normalized causal observations and the
preceding descending-neuron state: 1,697 → 128 → 128 → 1. It estimates return,
not rewards, and does not drive any actuator. Explicit task code adds rate rewards
for measured velocity/yaw tracking, upright support, and costs for tilt, vertical
motion, action changes and prohibited support. Rates are multiplied by 0.002 s;
a physical fall adds a one-time −1 and resets that world. Command tracking uses a
50 ms velocity filter to avoid equating natural within-stride yaw with command
drift. The original raw evaluation gates remain unchanged and visible.

Hold, slow/normal/fast walking and both turns coexist in the same world batch.
Episodes last up to two seconds. Timeouts bootstrap the terminal observation's
value; falls do not. Neither transition leaks value or recurrent memory from the
next episode. Chunk boundaries retain the actual preceding rollout state; memory
is cleared only at true episode resets. A short original-teacher-data rehearsal
batch follows each rollout, with the same held-out episode split, 8-step context,
25% reset contexts, MSE + 0.02 utility CE. Its compute and frame count are reported
separately and also included in total training wall time. This is an explicit
recipe change from pure imitation, not a continuation with identical optimizer.

Fourteen focused tests pass, including exact recurrent sampled-action replay,
physical-reward gradient flow into internal cells, timeout/failure return handling,
and native/single-body physical agreement. First run is a short implementation
pilot; a longer trial depends on its measured behavior and numerical health.

### Outcome PPO implementation pilot 01

The 20-second requested pilot completed one final rollout at **21.075403 s**,
collecting **30,720 physical transitions / 61.44 aggregate simulated seconds**.
Setup took **4.765655 s**. Collection (CPU physics + CUDA inference + bookkeeping)
took **15.905340 s**; PPO optimization **3.028985 s**; explicit rehearsal
**2.131341 s / 3,840 frame presentations**. There were 16 PPO minibatch updates
across 15 rollouts, 32 live worlds and 16 CPU physics threads, approximately
**1,458 transitions/s including learning**, with **5,342,384,640 bytes** peak CUDA
allocation. The training-only critic has 233,985 parameters. Physical-reward
gradients reached over 165,000 cells for each of excitability, leak and bias, with
finite nonzero gradients and measurable parameter changes. This proves gradient
flow through the measured graph; it does not yet prove useful brain causality.

Two collection worlds fell; most episodes had not yet reached their two-second
limit when this short run ended. Do not treat completed-episode counts as a
success denominator. That pilot retained failure metrics but not training-fall
trajectories. The next version stores the last up to 128 action frames (0.256 s)
for every training fall, with full physical states/actions/utility and a model
hash, and saves an error snapshot for numerical exceptions. Full six-case
teacher-free evaluation captures already retain all states, including failures.

The greedy pilot checkpoint stayed upright with valid support in all six cases,
but still failed the original raw tracking gates. Normal walking's 100 ms
block-mean forward error was **0.093 cm/s**, and yaw error **0.435 rad/s**. Hold
still drifted about **10 mm net in two seconds**; the right turn remained weak.
These are diagnostics, not integrated behavior acceptance.

The initial Adam learning rate 1e-5 caused KL excursions (first approximately
0.282, versus the 0.03 early-stop threshold), allowing only one or two PPO chunks
per rollout. For the next **300-second trial**, reduce only the actor/exploration
learning rate to **2e-6** and use seed 28002. Start again from the same DAgger01
parent, not from the pilot, preserving the pilot as a separate experiment. Retain
reward weights, recurrent architecture, 32 worlds, 16 threads, rollout/rehearsal
recipe and evaluation cases. This is an explicitly documented step-size adjustment
based on the measured update size. It has not changed the acceptance gates.

### Outcome PPO 01: timed five-minute trial and retained limitations

Starting again from DAgger01 with the declared 2e-6 step size, seed 28002 and
unchanged reward, the run completed **301.196457 s** of training wall time after
**5.697281 s** setup. It used 32 independent physical worlds, 16 native CPU
threads and the RTX 4090 for the shared graph actor/learning. Collection took
**142.704298 s**, PPO optimization **139.156288 s**, and explicit imitation
rehearsal **19.193652 s**. Rehearsal is included in the total, not hidden as free
training. There were **136 rollouts, 278,528 actual physical transitions,
557.056 aggregate simulated seconds, 1,090 PPO chunk updates and 34,816 rehearsal
frame presentations**. Peak CUDA allocation was **5,342,385,152 bytes**.
Checkpoint SHA-256:
`6983be3ccc477ecab3812e751c26e77311f93390c94830821ba8037aedb3449f`.

All six greedy teacher-free development cases remained upright with valid foot
support and zero MuJoCo numerical warnings. The original raw tracking gates
still failed in all six cases; the evaluator now exits 2 for that outcome.
Comparing the same recorded-state 100 ms diagnostic against the DAgger01 parent:

| Command | Forward RMSE before → after, cm/s | Yaw RMSE before → after, rad/s |
| --- | --- | --- |
| Hold | 0.452 → 0.488 | 1.080 → 0.880 |
| Slow walk | 0.190 → 0.155 | 1.437 → 0.787 |
| Walk | 0.103 → 0.124 | 0.498 → 0.476 |
| Fast walk | 0.163 → 0.135 | 0.535 → 0.570 |
| Left | 0.187 → 0.352 | 0.508 → 0.398 |
| Right | 0.989 → 0.369 | 2.706 → 0.701 |

This is a mixed improvement, not universal success. The right case became much
less erratic, but its mean yaw was only −0.183 rad/s for a −0.75 command; it did
not learn accurate right-turn tracking. Normal walking remained close to its
1 cm/s target (mean 0.927). Hold drifted **8.72 mm net in two seconds** and chose
rest in only **25.7% of frames**. An explicit forced-rest intervention on the same
checkpoint still drifted **8.66 mm**, with worse yaw oscillation. That diagnostic
is ineligible for policy acceptance and proves changing utility selection alone
is insufficient. The motor response must learn to arrest motion too.

There were 272 completed training episodes, 28 ending in a fall; the remaining
live partial episodes are not counted as completed successes. All 28 falls have
hashed last-128-frame physical traces in the ignored run directory. Training
exploration failures and deterministic deployment checks are distinct. The
checkpoint's ancestry includes **600.083779 s of supervised optimization** plus
this PPO stage; the failed DAgger02 and short PPO pilot are separate experiments,
not part of these weights. The inherited pretrained teacher's training is not
included in our times.

The next focused hypothesis is a dense stationary-motion cost when the command
is zero: the current narrow exponential gives very little distinction between
bad drifting holds. Test that while preserving moving-command rewards and explicit
walking rehearsal, then test actual walk-to-stop transitions without clearing
recurrent state. This hypothesis is not yet implemented or validated. Full
flight, recovery, learned survival utility and multi-agent arena integration
remain open; [flight prerequisites](FLIGHT_PREREQUISITES.md) record the inspected
upstream aerodynamic settings and direct-wing-control requirements.

### Complete-case physical/brain film

`student_ppo01_all_commands.mp4` shows all six original two-second development
cases, one checkpoint, at 1× with observer eye cameras and actual anatomical
recurrent-state maps. It is a **12-second, 600-frame, 1600 × 900, 50 fps** diagnostic
film, not the final survival demo. Rendering/encoding/full decode took
**49.475143 s**. No case or failed tracking label was omitted.

Frame review caught an infinite collision plane whose finite display patch ended
under the right-turn case. The next video version enlarges only plane display
bounds. A regression test drops a sphere beyond the former display edge and
compares 500 native steps: poses and contact forces remain identical with the
larger display. Original captured physics and the first video are retained. The
new overlay calls the failed gate “Raw tracking gate” to distinguish that measure
from posture stability. Fourteen actor/body/learning checks plus this rendering
physics-invariance check pass.

The corrected `student_ppo01_all_commands_v2.mp4` rendered and fully decoded in
**49.169889 s**, retaining the same 600 frames / 12 s / 50 fps / 1× case sequence
and checkpoint. Opening, walking, turning and final frames were visually inspected;
the visible ground now continues beneath the fly. The change is restricted to
replay plane display size and is recorded in each segment's metadata. All 28
training-fall trace hashes and finite state arrays were independently checked on
the training machine. The journal/index retain the original video as well.

### PPO 02 stationary-cost continuation (declared before training)

Continue the PPO01 checkpoint (`6983be3ccc477ecab3812e751c26e77311f93390c94830821ba8037aedb3449f`)
for a requested 300 seconds, seed 28003, 32 worlds / 16 physics threads. Retain the
actor, normalizer, critic, both Adam states and learned motor exploration scale.
The new resume path recognizes a PPO parent explicitly; behavioral-cloning Adam
has a different parameter layout and is not treated as PPO state. Learning rate
remains 2e-6, and rollout, recurrent chunks, task mixture and rehearsal are unchanged.

Add two bounded costs only when the entire command vector is zero, using the
existing 50 ms filtered physical velocity. Translation costs
`1.5 * speed / (0.25 + speed)` per second, with speed in cm/s. Rotation costs
`0.25 * abs(yaw) / (0.5 + abs(yaw))` per second, yaw in rad/s. They are subtracted
from reward rates before multiplication by the 2 ms action interval. They never
penalize a requested walk or turn-in-place. A new test verifies stationary drift
ranking, zero cost at rest, bounded costs and exactly unchanged moving rewards.
The original narrow tracking exponentials and all physical gates remain present.
This hypothesis needs a measured result; it is not an accepted improvement yet.

Afterward compare the fixed six development commands and a separate continuous
walk → stop → walk sequence. Command changes must retain recurrent memory and
physical state. This addresses a gap in separate reset-based hold/walk tests.

### Continuous command-transition evaluator

Added a separate development test with two seconds walking at 1 cm/s, two seconds
stopping, then two seconds walking again. Heading initializes at 0.1 rad. No body
or neural reset occurs at command boundaries; the report records actual physics
time, root position and preceding memory norm there. The original six fixed-command
gates remain unchanged. The new recorder can display the actual command per frame.

Transition gates are declared before evaluating candidates. After a 0.25-second
settling interval, moving phases require mean forward error below 0.25 cm/s,
lateral error below 0.25 cm/s and mean yaw error below 0.35 rad/s. Stop requires
late planar-speed RMS below 0.1 cm/s, 100 ms block-yaw RMS below 0.3 rad/s, late
drift below 1 mm and total stop-phase drift below 2 mm. Every phase must be complete,
upright above 0.5, height above 0.6 mm, and prohibited support below 0.1 body weight
at every physics substep. Two metric tests reject sustained drift, a late stop
after excess travel, incomplete duration and prohibited support while accepting
an idealized braking trace. Those tests validate scoring, not physical skill.

### Stopping reward result and online correction pilot

PPO02 retained the PPO01 actor, critic, normalization, exploration and optimizer
states. Its additional zero-command costs did not teach braking: the six-case
hold still moved approximately 8.55 mm in two seconds, and normal/right walking
tracking regressed. Keep it as a failed candidate; do not use it as the next parent.
A continuous walk–stop–walk test retained physical and neural state throughout.
PPO01 traveled 8.64 mm during stop; PPO02 traveled 9.14 mm. Both remained upright
with permitted support. Exact reports and checkpoints are retained separately.

A training-only intervention let the inherited walking teacher brake from PPO01's
moving state, with its stationary reference anchored to current xy and heading.
Stop travel was 0.677 mm total and 0.00870 mm after the settling interval. This is
not student success: the teacher executed the stop actions. The new raw planar
speed gate still rejected the teacher's within-stride jitter despite negligible
late drift; retain that result and examine block-averaged velocity separately.

Next pilot, declared before running: resume **PPO01**, seed 38001, 32 physical
worlds / 16 CPU threads, requested **60 seconds**, CUDA full-graph learning, new
imitation Adam at **3e-5**, eight-step recurrent chunks. This is supervised online
correction, not PPO. Carry the student's real recurrent state across chunks and
command changes. The inherited teacher supplies zero-command braking labels;
a frozen PPO01 graph actor supplies moving-command retention labels, both on the
student's current physical observations. Balance the two groups' action MSE and
add 0.02 utility cross-entropy (rest/explore). No teacher is deployed at evaluation.

For training execution only, blend 50% target / 50% student initially and fade the
target contribution linearly to zero halfway through the timed pilot. Half the
worlds switch command at one second without resetting body or memory; episodes
end at two seconds or a physical fall. Record all training falls, timings, source
hashes and internal-cell gradients. Keep the same graph architecture and walking
passive-appendage mask. Accept no improvement from loss alone: independently
check the saved student on fixed commands and continuous walk–stop–walk.

The batched braking adapter was compared with the inherited single-world oracle
on actual physical states and verified not to write poses. Native single/batched
physics agreement remains tested. Nineteen focused tests passed before this pilot.

### Online pilot outcome and next retention trial

The 60-second online pilot kept all six fixed cases upright with permitted support
and zero numerical warnings. Teacher-free hold moved approximately 0.803 mm,
compared with PPO01's 8.72 mm. In the continuous test, stop travel was **1.314 mm**
and late drift **0.229 mm**, without clearing recurrent state. Mean late forward
velocity was 0.0156 cm/s. However, normal walking regressed: continuous forward
speed was only 0.341 cm/s initially and 0.374 cm/s after resuming, with substantial
yaw drift. All original raw gates and the overall transition test still failed.
This demonstrates learned braking with incomplete skill retention, not a solution.

The inherited braking diagnostic's 100 ms block-planar RMS is 0.00451 cm/s while
its raw RMS is 0.165 cm/s. Preserve both: an instantaneous-speed gate can reject
stationary contact vibration. This observation does not excuse walking drift.

Next run declared before execution: return to PPO01 weights, same seed family
(seed 38002), 32 worlds / 16 threads, same 3e-5 new imitation Adam, eight-step
chunks, command mixture, targets and 0.02 utility CE. Increase moving retention
loss weight from one to **four** and request **300 seconds**. The original 50%
training-only assistance still fades to zero halfway through the timed run.
The frozen retention actor remains PPO01. Independently test the final single
student, with no inherited or frozen teacher present. The pilot remains separate
and is not part of this new checkpoint's ancestry.

The online probe ran **60.123372 seconds** after **9.464741 seconds** setup:
**45.190153 seconds** collection plus supervised forward computation and
**14.921138 seconds** backward/optimization. It collected **63,488 physical
transitions / 126.976 aggregate simulated seconds**, in **248 updates**, using
32 native CPU physics worlds and CUDA graph learning. All **29** saved training
fall traces were hash-checked and finite. The pilot checkpoint is
`97f2877504cced0d8265255f9806d9bd493630b5d8665dbffc6c2f4205b7ea35`.
PPO02's **14** saved falls were also verified. The pilot is explicitly imitation,
not additional physical-reward PPO training.

### Flight body preparation

Implemented a separate full-anatomy flight preset with upstream ellipsoid wing
fluid parameters, 20 kHz physics, 5 kHz control and unfiltered wing torque channels.
All 78 actuator outputs remain; the input schema exposes effective actuator input
so removing six wing filter states does not remove observation channels. Existing
walking physics and observations retain numerical agreement in regression tests.
This is a physical preparation step, not permission to run walking weights at ten
times their learned recurrent rate.

A short, explicitly scripted aerodynamic probe starts airborne and compares air,
vacuum and a halved timestep. It supplies only bounded wing controls; no body pose
writes after initialization or root-force injection. The upstream synthetic wing
pattern generates about 28% of body weight upward force, insufficient to hover.
All three trajectories remain finite without warnings. Preserve this failure to
support body weight; stronger/learned wing control is still required. Exact settings
and limitations are in FLIGHT_PREREQUISITES.md. Twenty-one focused tests pass.

### Online correction 01: walking retention improved, failures remain

The declared fourfold-retention run completed **300.229040 seconds** of training
with **9.815098 seconds** setup. It collected **321,280 physical transitions /
642.56 aggregate simulated seconds**, across 32 native CPU worlds and 16 threads,
with full-graph learning on the RTX 4090. Collection and supervised forward work
took **225.503499 seconds**, backward/optimization **74.220068 seconds**. There
were **1,255 updates**, **159 command changes without resets**, and **324 completed
training episodes / 26 falls**. Every fall trace was hash-checked and finite.
Peak CUDA allocation was **2,995,016,704 bytes**. All internal-cell parameter
families received finite gradients and changed.

This student uses checkpoint
`9b9ab703483a52d06eb1700172e762b8226e3b4b6597678176a89bd886b6649a`.
It inherits PPO01 directly, excluding the one-minute pilot and failed PPO02 from
its ancestry. Its training targets came from the inherited braking teacher and
frozen PPO01 walking actor; both were removed for evaluation. This is supervised
online correction, not an additional PPO stage.

Continuous teacher-free walk–stop–walk stayed upright with permitted support.
Mean walking/resumed forward speeds were **0.951 / 0.922 cm/s** for a 1 cm/s command.
Stop travel was **1.113 mm** total and **0.0609 mm** after settling, versus PPO01's
8.64 mm / 7.55 mm. The same memory and physical body persist across both command
changes. Stop's 100 ms block-planar RMS was **0.01168 cm/s**; the retained raw RMS
was **0.13379 cm/s**, still failing the raw 0.1 gate. Walking yaw means
**0.372 / 0.389 rad/s** also exceed the declared 0.35 gate. Report the overall test
as failed, not solved by these more encouraging individual measures.

Five of six fixed-command cases kept upright, permitted support; the **slow case
toppled**, with prohibited support reaching 13.13 body weights. All original raw
tracking gates failed, with zero numerical warnings. Normal walking's 100 ms
forward RMSE improved to 0.060 cm/s. These development results establish useful
walking/braking retention while exposing slow-speed and heading requirements.

Recorded all six fixed cases plus the complete continuous sequence, one checkpoint,
at **1× / 50 fps / 1600 × 900 / 900 frames / 18 seconds**. Render, encode and full
decode took **104.858421 seconds**. The source review manifest lists both original
capture reports and hashes; no case was omitted. Local full decode and twelve
sampled frames verified the copied video, including the slow fall and both command
changes. It retains actual anatomical latent activity and observer eye views.

The clean-source flight physical probe (source ecd797d) reproduced the initial
three air/vacuum/timestep trajectories exactly by state hash. That is a force
prerequisite, not learned flight. Walking/flight integration, accurate commands,
trained survival needs, brain-contribution tests and the multi-agent final film
remain open. The next iteration must preserve current stopping and walking while
addressing failures; it must not present separate checkpoints as one solution.

### Diagnose slow walking and declare next physical-outcome pilot

Replay analysis of online01 shows the first upright failure at **1.742 s** in
slow walking. Rest was not selected in the 1.0–1.8 s windows leading to the fall;
the actor continued choosing explore. This points to motor dynamics rather than
an unwanted stop decision. Normal walking chose explore for all 1,000 frames;
hold chose rest for 999/1,000. Replay analysis is diagnostic, not a new rollout.

Next declared pilot: start from online01's retained single actor, request **60 s**
physical-outcome PPO, 32 worlds / 16 CPU threads, seed 28004, learning rate 2e-6,
initial Gaussian motor scale 0.04, original 64-step rollouts / eight-step chunks /
two PPO epochs / original-data rehearsal. Retain the bounded stationary costs
1.5 for translation and 0.25 for yaw to discourage loss of the new hold. Since
the parent is an imitation checkpoint, initialize a new training-only critic,
PPO optimizer and exploration state; inherit only actor and normalization. Keep
all six development cases and the uninterrupted transition check. This trial
is not accepted on reward alone and does not replace online01 automatically.

The already downloaded official flight dataset includes its wing kinematics.
Added an explicit pattern-file option to the force diagnostic. Under the same
20 kHz physics, its supplied pattern produces 0.555 body-weight mean upward force
versus the synthetic pattern's 0.276, and saturates wing torque on 5.78% of samples.
Neither holds altitude. At 40 kHz the supplied pattern gives 0.596 body-weight
force; the larger timestep sensitivity must be retained. Flight teacher export
now labels its actual source directory rather than hard-coding "walking"; no
teacher training or student deployment is implied by conversion.

### PPO probe 02 and verified flight-teacher import

The next physical-reward pilot kept all six two-second fixed-command bodies
upright, including the formerly toppling slow case, but the slow case still
approached the stability threshold (minimum upright 0.654) and tracked poorly.
The continuous check retained a small stop displacement but added substantial
yaw during stopping; resumed forward speed fell to 0.593 cm/s. This is a mixed,
unaccepted candidate, not a replacement for online01. Preserve every original
failed gate and report. A curriculum gap is now explicit: PPO rehearses fixed
commands, whereas online correction included actual command switches. Future
physical-reward refinement must retain those transitions as well as steady walking.

The official flight SavedModel uses a 104 → 256 → 256 → 256 → 12 mean network,
with LayerNorm/tanh first and ELU hidden layers. Its 12 saved parameter tensors
have a different layout from the walking teacher's 14. The generic importer
supports both and validates each against exported TensorFlow reference outputs.
Flight maximum absolute discrepancy on 32 saved probes is **8.6427e-7**; walking
retains its previous checked values. No teacher retraining occurred.

The flight adapter maps the complete body's observed state to the teacher's
25-joint/104-input schema. The inherited policy supplies 11 physical commands plus
a wing-frequency control; the upstream supplied wing pattern converts the latter
into six bounded wing commands. Remaining leg position servos receive the
upstream retracted spring pose as a training-only target, with all limbs and
contacts still physical. Runtime student acceptance must contain neither this
teacher nor the wingbeat generator. Joint transmission type is checked before
setting posture targets, avoiding accidental interpretation of adhesion body IDs
as joint IDs. Tests check no live pose writes, no root forces, bounded actions,
zero adhesive posture targets, and both independent numerical export references.

The first 0.2-second inherited-controller hover stayed close to its 10 mm target
height, with 0.192 mm root-position RMSE and no ground loading or solver warnings.
This was a development probe before the transmission-type guard; retain its exact
report. Longer clean-source hover and moving-flight probes are required before
using this expert for student demonstrations. **23 focused tests pass.**

Flight teacher/pattern attribution is in assets/embodied_fly/teachers/FLIGHT_NOTICE.json.
The official dataset metadata was checked directly against Figshare's article API;
the existing flight dataset ZIP matches its published MD5. These GPL-3.0-or-later
teacher/data assets are distinct from the anatomical model's source license.

### Longer clean-source flight-teacher checks

Source `08492b4` ran the complete body with the inherited expert, supplied wing
pattern, bounded actuators and preserved floor contacts. **One-second hover**
maintained the 10 mm target height, with **0.174 mm** root-position RMSE and a
minimum height of **9.952 mm**. Setup took **2.776268 s**, stepping/capture/state
writing **3.504010 s**. **Half-second forward flight at 20 cm/s** traveled to
99.905 mm forward while retaining altitude, with **0.319 mm** root-position RMSE
and minimum height **9.769 mm**. Setup took **1.827285 s**, stepping/capture/state
writing **1.771731 s**. Both had zero numerical warnings and zero prohibited
ground loading. The free root was initialized once; all later motion came from
actuation, gravity and fluid forces.

These checks establish a useful training expert and a functioning complete-body
flight environment. They do not establish student flight, takeoff or landing.
The next integration question is explicit control timing: preserve the shared
actor's walking/stopping skills while adding 5 kHz wing control. The existing
500 Hz walking weights cannot simply be called ten times faster and declared
transferred. Check full-flight physics with the existing policy clock, then
validate a documented recurrent timing/curriculum scheme before accepting flight.

PPO probe 02's measured training time was **61.072358 s**, setup **4.956591 s**,
with **55,296 transitions / 110.592 aggregate simulated seconds**, **222 PPO
updates**, and **6,912 rehearsal frames**. Collection took **29.157640 s**, PPO
optimization **28.160629 s**, and rehearsal **3.733163 s**. One of 32 completed
training episodes fell; its retained trace was verified. Checkpoint
`f8459aeb48997f79f81f7cb5bfd7433a4d0b1c53a2b98c8fe21af4f79441de7e`
remains diagnostic because the continuous resume regressed. Online01 remains the
preserved walking/braking reference; neither is the final survival controller.

### Walking-to-flight physics compatibility, declared before evaluation

Before changing neural timing, evaluate online01 in the full flight physical
preset at its original **500 Hz actor clock**. Each action is held across ten
0.2 ms motor intervals, with 20 kHz physics and every-substep contact/warning
checks. The graph weights, neuronal update timing, original six commands, two-second
durations and development seed remain unchanged. Report the physical preset,
physics rate, actor rate and motor interval independently. This is a compatibility
probe, not learned flight or acceptance of the faster neural clock.

A regression check compares one 2 ms held action with ten explicit flight motor
steps and obtains exactly equal physical states and observations. Invalid
fractional motor intervals are rejected without advancing the body. The original
walking stepping path still uses one 2 ms interval. Seven focused body/flight
checks pass before this evaluation.

### Explicit recurrent clock and measured physical transfer

The unchanged online01 actor at 500 Hz remained upright on permitted distal-foot
support in all six two-second flight-physics cases (source `65ffef7`). Zero solver
warnings. All original tracking gates still failed; normal-walk 100 ms mean speed
error increased to 0.488 cm/s and yaw error to 1.426 rad/s. This preserves physical
control but does not establish matched walking quality. Report retained under
`runs/motor_online01_flight_physics_500hz`.

The next timing diagnostic is declared before execution: same online01 weights,
same flight model, same predetermined hold/walk initial headings, two seconds each,
now at 5 kHz actor rate. Four internal updates remain per action. A cell's leak
becomes `1 - (1 - leak) ** (actor_dt / 0.002)`, preserving its held-target relaxation
time. Legacy 500 Hz arithmetic is exactly unchanged. This is exact only for a
held neural target, not a guarantee of recurrent trajectory or behavioral
invariance. Unit checks cover numerical relaxation, gradients and invalid clocks.
Utility and sensory feedback are also sampled faster, so physical evaluation is
required before calling this transferred. Neither timing probe is flight learning.

Replay now samples actual recorded timestamps, with explicit metadata fallback
for older 500 Hz recordings. Independent tests verify one second yields 50 video
frames at either 500 Hz or 5 kHz capture, and reject mislabeled rates. Inherited
flight captures are labeled teacher + wingbeat generator, not MaleCNS student.
