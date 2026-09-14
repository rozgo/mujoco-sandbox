# Learning journal: from a utility selector to an embodied neural controller

This is the permanent account for future demos, videos and technical reporting.
Keep failed hypotheses as well as successful runs. The current project is in
progress; its complete flight/survival curriculum has not passed acceptance.

Latest reference-only course correction: the user rejected jerky hover despite
the old numerical gate passing. [Per-tick wing forces](INSTANT_WING_FORCES.md)
remove the extra activity average, and the [PID reference](PID_HOVER_REFERENCE.md)
uses 1,000 Hz physics. Its ten-second capture has a 0.13 mm settled height band and
0.16 mm settled peak position error, with a 0.60 mm cold-start peak error retained
in the video. Eleven brief controller probes and a committed-source capture are
archived; no policy training occurred. A fixed camera and live error readouts
supported visual review. The user accepted `pid_hover_reference_v1.mp4` on
September 13: "hover looks great." The learned checkpoint remains intact.

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

The 5 kHz clock-transfer probe failed (source `c9fed6f`): hold remained upright
but developed prohibited support at 0.455 body weight; normal walking toppled,
with 6.923 body-weight prohibited support. Both had zero numerical warnings.
Cell relaxation scaling alone is insufficient to preserve the walking behavior;
do not replace the retained 500 Hz walking execution with this failed transfer.

Next declared pilot: offline mixed-clock imitation in the same actor, initialized
from online01. Original walking demonstrations retain their 500 Hz clock; eight
new 0.3-second inherited-flight demonstrations use 5 kHz, speeds 0/5/10/20 cm/s
and seeded initial wing phases. Choose each clock with equal probability per
homogeneous minibatch. Use 32 neural sequences, 16 burn-in + 16 supervised steps,
25% reset-start batches, 60 seconds optimization, learning rate 1e-4, and an extra
unit-weight six-wing MSE only on flight batches. Retain parent normalization and
Adam state when available. Validate each clock separately. All neuron dynamics
and interfaces remain in one actor/checkpoint; fixed connectivity is unchanged.
These are offline neural sequences, not parallel live physics worlds. This stage
has no altitude/visual inputs yet and tests initial direct-wing learning only.
A deployment clock transition and recovery of terrestrial behavior on the shared
flight physical model remain open, as do takeoff, landing and survival utility.

The evaluator for this pilot loads only the first physical state from a declared
expert capture, then runs the student with all 78 bounded actuators and no teacher
or oscillator calls. It measures short airborne tracking against a declared
1 mm root-RMSE / 8 mm minimum-height envelope, preserving failure trajectories.

### First direct-wing learning result and diagnosed information loss

Source `904b8f3` collected eight complete physical flight demonstrations in
**20.218079 s**, after **4.194998 s** setup: 12,000 actuator transitions / 2.4
simulated seconds. All eight met the predeclared training envelope with zero
warnings or prohibited support; all local copies passed hashes, finite-state,
clock and shape checks. The inherited teacher alone generated these examples.

The mixed-clock full-graph pilot trained for **60.168454 s**, setup **3.645980 s**,
validation **0.597049 s**, on the RTX 4090. Its 32 offline neural sequences gave
**207 updates / 105,984 supervised examples**, with 105 walking and 102 flight
batches. Peak CUDA allocation was **5,176,925,696 bytes**. All three internal-cell
parameter groups received finite nonzero gradients and changed. Checkpoint
`ecfdcf3e2633639e03d0f0fc94da46c727705780dd46761bc06ec8974986496c`
is archived as `motor_flight_probe_01.pt` and **not promoted**.

Flight imitation validation improved 0.463187 → 0.019442, while walking validation
changed 0.017634 → 0.016919. Nevertheless, both held-out-initial-state airborne
probes lost altitude: hover root RMSE 7.382 mm, forward-flight RMSE 15.462 mm over
0.3 seconds. The same checkpoint lost stability in hold, normal walking and right
turning; its continuous stop toppled and resume did not recover. All raw tracking
gates and failure captures remain archived. No numerical warnings occurred.
This demonstrates that low supervised error on teacher states is insufficient
for closed-loop control; online01 remains the prior walking/braking reference.

A clean-source replay diagnostic (`d10a919`) compared the first 30 ms at identical
initial physical pose. The student's wing excursions were larger, but its sampled
mean upward passive force was only **0.0900 body weight**, versus the teacher's
**0.9630**. These are 150 control-boundary samples, not physics-substep averages.
The student crossed below the 8 mm flight envelope at **21.6 ms**.

Crucially, wing yaw/pitch speeds exceed the inherited 1,000 rad/s observation clip
in about 41–55% of teacher samples. The walking-trained normalization followed by
encoder clipping saturates **90–100%** of all six wing-velocity channels in this
window. We have identified information loss, not proved it is the only cause.
Before extending the failed training, add continuous wing-speed feedback with an
explicit observation-schema migration that preserves the old actor's outputs at
initialization; retain the old capture/reproduction path. Ground retention should
also use the preserved student's actual walking/stopping behavior, because
original-teacher MSE alone failed to preserve those skills. Re-evaluate physical
wingbeats and ground behavior after the correction rather than trusting MSE.

The new one-second inherited hover video is accurately labeled **reference
teacher + wingbeat generator**, with no MaleCNS learner. Rendering took
**3.843421 s**; all 50 frames decoded at 1600×900 / 50 fps / 1×, with visual checks
at the start, middle and end. It is a reference for training, not student success.

### Wing-speed observation repair and retained-student rehearsal (declared)

Add six continuous wing-joint velocities scaled by 2,000 rad/s to the original
383-input observation. Keep the original 383 values, normalization and encoder
matrix intact. A zero-initialized 6→128 sensory contribution feeds the existing
sensory hidden layer, then the same graph/utility/motor path. It is not a separate
controller and has no direct motor bypass. The migrated actor initially has
identical actions, utility and recurrent state in a 20-step small-graph test;
new sensor weights receive finite nonzero motor-loss gradients. The old snapshots
and runtime observation path remain reproducible.

Collect eight predefined retained-student episodes at 500 Hz: hold, slow/normal/
fast walking, left/right turns, and two continuous walk-stop-walk trials. Seed
44001 determines headings. Keep failures and reject whole unstable/prohibited-
support episodes from rehearsal. Labels come from the retained online01 actor's
actual live observations and executed actions, not the inherited walking teacher.
Command changes do not reset memory or body state.

The next pilot starts again from online01, not the rejected mixed-flight weights.
Use the new retention corpus and existing eight flight demonstrations, with equal
clock-group sampling, 32 neural sequences, 16 burn-in + 16 supervised steps,
25% reset-start batches, 60 seconds, seed 45001, learning rate 3e-5 and unit extra
wing MSE on flight batches. The observation extension preserves the parent's
initial function; Adam is explicitly initialized fresh because parameter layout
changed. Existing normalization is retained, with mean 0/std 1 for the six new
scaled measurements. First verify full-graph migration on CUDA, then train and
measure physical flight and ground retention. No outcome is assumed from MSE.

The first full-graph CUDA migration check retained a small numerical difference:
maximum action error 4.18e-7 and latent-state error 1.01e-5 over 64 steps × eight
sequences at both clocks. The strict equality check failed and its report is
preserved. The extended tensor passed a strided 383-column view into the original
linear layer; explicitly restoring the original contiguous matrix layout is the
next check. Do not silently weaken the exact-migration assertion. Separately,
all 12,000 flight frames have zero encoder clipping in the six new channels
(maximum scaled absolute speed 1.775 versus the safety bound 10), with identical
legacy observation values. This verifies information preservation, not flight.

The contiguous-layout check still found ~4.2e-7 maximum action differences. A
component probe found identical raw inputs, normalized inputs and first-layer
outputs, but **the unchanged parent repeated on identical inputs also differed
by 2.38e-7 in actions**. CUDA sparse execution is not bit reproducible here; the
zero adapter preserves the function, not a universal bit-level CUDA guarantee.
Retain both failed strict reports. Before retrying, revise this *numerical
migration* check to action/utility absolute tolerance 1e-6 and recurrent-state
1e-4 over 64 steps, and report the unchanged parent's own repeat-run variation
alongside it. The original physical task gates are unchanged. This explicitly
corrects the earlier overly strong exact-CUDA assertion; it is not a training
or physical-success claim.

Pilot 02 completed 60.094962 seconds of optimization with the repaired inputs and
retained-student corpus. Flight imitation MSE fell 0.463187 → 0.052054; ground
retention MSE rose 0.000421 → 0.002731. All six new velocity channels remain
unclipped. Five of six fixed ground cases and the continuous transition stayed
upright with permitted support; tracking is degraded and the slow case fell.
Both airborne trials still fell. The first 30 ms of hover replay shows only
0.0526 body-weight upward passive force and substantially slower wing motion
than the expert. This checkpoint is a diagnostic, not a promoted controller.

Allow one bounded **180-second continuation** from pilot 02, preserving its Adam
state, learning rate 3e-5, dataset splits, clock mixture, 32 sequences and 16/16
burn-in/supervision. Seed 45002. The last flight-weighted losses were still falling
(0.270 at 31 s, 0.217 at 41 s, 0.198 at 52 s), so this tests insufficient fitting
before changing the method. Evaluate actual wing motion, both airborne cases,
all ground commands and continuous transitions afterward. Do not infer success
from lower loss or silently promote this candidate. The prior reference remains
online01; a failed continuation should motivate corrective/on-policy learning,
not indefinite extensions of this same warm start.

### Three-minute continuation and corrective flight collection

The declared continuation (`motor_flight_01`, seed 45002, source `71b48e8`) ran
180.173966 seconds, 608 updates and 311,296 supervised examples. Adam resumed.
Flight validation MSE improved 0.052054 → 0.017615, but both 0.3-second airborne
probes still fell. Five fixed ground cases remained stable; hold fell, and the
continuous walk-stop-walk trial toppled. Zero numerical warnings. Preserve this
checkpoint as rejected; lower teacher-state error did not improve physical skill.

Next collect eight 0.3-second flight episodes using **pilot 02** (the candidate
that retained continuous ground stability), seed 46001, **15% student / 85%
teacher** bounded actuator mixtures. At each actual visited state the inherited
teacher supplies a correction label. Save teacher, student and executed actions
separately; recurrent state persists within each episode. The teacher alone has
its reference and wingbeat generator. This is supervised data collection with
training assistance, not teacher-free flight or PPO. All failures remain captured
and whole failed episodes are ineligible for imitation, as before. Start with a
small mixture because the current unassisted student loses height within tens of
milliseconds. Validate the mixture physically before training on it.

A diagnostic over the 0.3-second held-out hover capture also found inherited
wing-position normalization saturates roughly 62–84% of angle samples. The new
velocity inputs are continuous. This identifies another information limitation;
we have not yet established its causal contribution or changed the angle schema.

If the correction collection yields at least four complete eligible episodes,
run one **60-second corrective imitation pilot** from pilot 02, seed 47001,
learning rate 3e-5, 32 parallel neural sequences and 25% reset-start batches.
Rehearse retained online01 ground data, original flight data and the new corrected
flight data. Keep equal sampling of ground/flight clocks. Increase the ground
loss multiplier to **4** to protect its earlier behavior; this is a declared new
retention setting, not the previous recipe. Use **64 burn-in + 32 supervised
steps**: at 5 kHz the supervised window spans 6.4 ms, longer than a nominal
218 Hz wingbeat, and the burn-in gives recurrent state 12.8 ms of history.
The earlier 16-step supervised window covered only 3.2 ms. This combined pilot
tests corrective data plus longer context and stronger retention; it cannot
isolate which change causes any improvement. Evaluate the same teacher-free
airborne and ground cases afterward. No extra deployed controller is added.

The assisted collection completed all eight episodes with no physical/numerical
failure: 45.966942 seconds collection, 5.629544 seconds setup, 12,000 physical
transitions / 2.4 simulated seconds. Every capture hash, timestamp, finite state,
actuator mixture and causal previous-executed-action input was checked on both
machines. This is teacher-assisted data, not learned flight.

The corrective pilot ran 60.676444 seconds, 82 updates and 83,968 supervised
examples. Both airborne tests still fell. Five fixed ground cases stayed upright;
normal walking fell. Continuous walk and stop stayed stable, with 0.0246 mm late
stop drift, but resume toppled. Preserve all failed gates and keep online01 as
the walking reference. The changed data/context means absolute MSE should not be
compared directly to earlier pilots. No candidate was promoted.

The clean-source angle audit over the first 30 ms of expert hover confirms
**64–85.3% wing-angle encoder clipping**. New wing-speed feedback remains unclipped.
Next extend the same sensory layer with six continuous angle measurements, while
preserving the existing velocity extension and original 383-input path. Use
zero-initialized extra columns, verify full-graph numerical migration, and verify
single/batched physics observation agreement before a bounded learning pilot.
This is an input-information repair, not evidence that flight will then succeed.
Avoid further identical-data extensions without a new physical result.

### Continuous wing-angle repair (declared before learning)

Extend the six existing wing-speed inputs with six measured hinge angles / pi,
for **395 inputs**. Keep the original 383 observations, their normalization,
the existing velocity measurements, and their learned encoder columns unchanged.
Expand the same sensory-input matrix from 6→128 to 12→128, copying its trained
first six columns and initializing the added columns to zero. No new recurrent
controller, motor bypass, oscillator or privileged phase input is introduced.
Native batched learning, individual evaluation, corrective collection and dataset
augmentation must all use the same schema; legacy checkpoints stay reproducible.

Verify a 20-step small-graph migration, gradients into the new columns, blocked
sensor-to-motor influence when graph edges are removed, single/batch physical
agreement, captured-data input agreement, and full-graph CUDA migration at both
clocks. Retain the previously declared numerical tolerances and record unchanged-
parent repeat differences. Check all new velocity/angle channels for clipping.

The bounded angle-input pilot starts from **motor_flight_probe_02**, the last
flight warm start that retained a stable continuous ground transition. Use the
same three corpora as corrective pilot 01, 32 sequences, 64 burn-in / 32 supervised
steps, ground multiplier 4, wing multiplier 1, equal clock-group sampling, 25%
reset starts, learning rate 3e-5, **60 seconds**, seed **48001**. Adam initializes
fresh because its sensory parameter shape expands. This optimizer difference is
explicit; the pilot is not a strict one-variable comparison. The walking reference
remains online01. Evaluate actual unassisted wing motion, both original airborne
cases, all fixed ground commands and continuous transitions before promotion.

The 395-input migration passed both full-graph CUDA checks (maximum action
difference 4.77e-7, within the previously declared 1e-6 tolerance). The complete
36,000-frame corpus has zero clipping in all twelve added channels; original
389-input values remain identical. The 60.494686-second angle pilot preserved
stable permitted support in all six fixed ground tests and the continuous
transition. Its walk/resume phase gates pass; stopping and all original raw
fixed-command gates still fail. Both airborne tests fell, with only 0.0547
body-weight sampled upward passive force in the first 30 ms. This is useful
ground stability evidence, not learned flight or final acceptance.

On 12,000 original expert observations, sensory preactivation RMS is 2.96219
from the original encoder, 0.005632 from added velocity inputs and **0.000656**
from the new angle inputs. Angle-column L2 is 0.02247 after training. These values
do not establish causal motor importance, but motivate testing faster adaptation
of the newly introduced input matrix. The angle information is available and
unclipped; its learned input contribution remains very small.

Next run **60 seconds**, seed **49001**, from `motor_flight_angle_probe_01`, using
the same data, clocks, 32 sequences, 64/32 context, loss weights and reset mixture.
Keep learning rate **3e-5** for the shared actor; give only the **12→128 sensory
extension matrix** multiplier **100** (3e-3). Preserve Adam moments and step
counts while moving that existing Parameter into its own optimizer group. A
test verifies every moment before the first step and after checkpoint resume.
No parameter is added and the measured graph stays fixed. This bounded rate
change tests whether the new feedback can become useful quickly; it does not
assume a flight improvement. Keep the angle pilot and online01 reference intact.

### Faster sensory adaptation failed; move to physical flight reward

`motor_flight_sensor_fast_01` trained for **60.607377 s**, 83 updates and 84,992
supervised examples. Setup took 6.295566 s and validation 1.587431 s. Adam state
was preserved. The added angle-input preactivation RMS grew from 0.000656 to
0.036742; flight validation MSE decreased to 0.035415. Nevertheless both airborne
tests fell, ordinary walking fell, and every continuous-transition phase became
unstable. The checkpoint is retained as a rejected experiment. More imitation
accuracy is insufficient evidence of physical flight.

The preserved angle01 candidate was captured again with actual neural projection:
all three six-second walk/stop/resume phases remained stable with valid support;
walk/resume passed their phase gates and stopping failed (0.8723 mm late drift).
This separate capture differs slightly from the earlier one because sparse CUDA
reductions are not bitwise deterministic. Its six-second 1× film fully decodes
to 300 frames, 1600×900 at 50 fps. Anatomy, signed latent-state colors, observer
eyes and raw failed gates are visible; the video is a development artifact.

Next use physical-outcome PPO for direct airborne control. Batched MuJoCo now
supports the existing complete flight preset: 20 kHz physics, 5 kHz control,
four physical substeps, all 78 motor channels and 72 filtered activation states.
Effective activation observations match native single-body evaluation exactly.
Driven-wing regression tests compare poses, velocities, aerodynamic passive
loads, observations, contact parameters and repeated partial resets. Walking
defaults remain unchanged.

Declare a **60-second** PPO pilot, seed **50001**, from angle01: 32 native CPU
worlds / 16 threads, RTX 4090 graph learning, 128-step rollouts, 32-step recurrent
chunks, two PPO epochs, learning rate 5e-6, exploration standard deviation 0.04,
60 ms episode timeout. Airborne resets use only frame zero of the six training
episodes in `flight_demonstrations_01`; validation episodes 4 and 6 remain excluded.
No teacher, oscillator, imposed wing phase or root force acts during rollouts.
The existing graph actor receives the same 395 causal inputs and emits all 78
commands. Its neural clock and PPO discounts scale to the 0.2 ms motor interval.

Physical reward rates favor commanded horizontal velocity, small vertical speed,
reset-height retention and reset-orientation retention, with a terminal penalty
below 8 mm or thorax upright below 0.5. Multiply rates by actual elapsed physics
time. Avoid an action-smoothness cost that would punish necessary wing flapping.
One measured ground imitation batch per rollout rehearses the training split of
`retention_online01_01` at its original 500 Hz. This is shared-actor continuation,
not separate walking and flying policies. Physical results must still pass the
unchanged airborne and ground evaluations; reward alone cannot promote it.

The first flight PPO pilot completed **62.569215 s**, 65,536 physical transitions
and 32 PPO chunk updates. All 892 completed airborne episodes failed in 6.2–22 ms.
Physical-reward gradients reached all trainable internal-cell parameter groups,
but both excluded-initial-state flight evaluations fell. Six fixed ground cases
remained stable; continuous resume toppled. Do not promote or extend this run
merely because its training path executes. Full results and trace hashes are in
[the PPO report](runs/motor_flight_ppo_probe_01/SUMMARY.md).

Next diagnostic: replay complete causal expert histories through angle01 with
frozen weights, read its actual motor-cell states, and fit a fixed-regularization
linear readout using only training episodes. Compare excluded episodes 4 and 6
against the current motor decoder and a readout using measured wing angles and
velocities. Keep the regression diagnostic-only; no fitted coefficient is added
to the deployed actor and no body is controlled. Expert previous actions remain
part of those recorded histories. Linear decodability can identify recoverable
information, but cannot establish autonomous oscillation, closed-loop flight,
or biological neural causality. Avoid drawing those broader conclusions.

The frozen angle01 diagnostic took **11.054809 s** (8.466627 s graph replay),
using eight independent neural sequences and all 12,000 expert frames. On the
two excluded episodes, current wing-output MSE is **0.093234**. A linear probe
of the 815 motor-cell states achieves **0.008276** after output bounding; a probe
of measured wing state reaches **0.007726**. This shows useful information is
linearly recoverable from the core under those histories. It neither changes
the deployed actor nor proves closed-loop flight.

Next inspect/cache the existing 256-unit motor hidden layer, then calibrate only
the existing final layer's **six wing rows and biases (1,542 parameters)** for
**10 seconds**, seed **51001**, Adam 0.003, minibatches of 1,024 cached training
frames. The encoder, intrinsic core parameters, utility head, motor hidden layer
and other 72 output rows remain bitwise unchanged. The graph continues to be the
only runtime path from observations to motors. This stage uses no extra deployed
module and introduces no scripted wingbeat or sensor bypass. Cache and checkpoint
hashes must match; whole-episode validation stays excluded from updates. Verify
all unchanged parameters and every non-wing output row, then evaluate actual
flight and ground behavior before promotion. This is targeted output calibration,
not another claim that offline error establishes flight.

The existing hidden layer also retained linearly decodable wing information
(bounded probe MSE 0.006600). Ten-second wing-output calibration reduced the
actual decoder's excluded-episode MSE to **0.006163**, with exactly 1,536 weight
elements and six biases changed. Both unassisted flight tests still fell.
Sampled first-30-ms upward passive force increased to **0.2306 body weight**,
but wing angles overshot the expert envelope; stable flight is not established.
See [the complete calibration report](runs/motor_wing_readout_01/SUMMARY.md).

Next collect teacher-assisted corrective trajectories using `motor_wing_readout_01`
as the student, beginning with 25% student / 75% teacher executed actions in eight
0.3-second episodes. Record actual executed feedback and teacher targets separately,
preserve failed collections, and exclude whole validation episodes. Cache the
same frozen graph's motor hidden features from both original and corrective
histories, then update only the existing six wing output rows in a bounded fit.
This addresses states the student visits while preserving its current architecture.
No teacher assistance may appear in the later acceptance rollouts. Retain the
angle01 ground video, all rejected pilots and the original online01 reference.

Corrective continuation is now fixed as seed **52001**, eight 0.3-second episodes,
25% `motor_wing_readout_01` / 75% teacher executed controls. Keep all failed
episodes in the collection manifest. If the accepted demonstration envelope is
not maintained, diagnose the physical failure before using those labels. Subsequent
readout fitting uses equal sampling of original and corrective training corpora,
with validation episodes excluded separately for each corpus. Start from the
same readout01 checkpoint, keep its entire upstream actor frozen, and calibrate
the same 1,542 existing wing output parameters for **10 seconds**, Adam 0.003,
1,024-frame minibatches, seed **53001**. This is supervised corrective training;
neither assisted collection nor low validation error establishes student flight.

All eight corrective episodes passed their collection envelope (12,000 frames,
2.4 simulated seconds; **6.392281 s** setup, **46.319653 s** collection). Captured
actions, exact 25/75 execution mixtures, previous-action feedback, timestamps and
state/model hashes were verified. Two frozen-feature replays took 12.025874 s
and 12.198033 s. The mixed readout fit used **10.000270 s**, 19,163 updates and
18,000 distinct training frames. Excluded original/corrective MSE changed from
0.006163/0.014430 to 0.006403/0.008380. Both unassisted flights still fell.
Continuous ground transitions stayed stable; slow fixed-command walking fell.
As in the preceding output-only fit, non-wing parameters are unchanged; retain
the observed outcomes without claiming a learned change in ground control.

The compiled wing joints do have angular limits, but compliant stops allow large
overshoots. Investigate an explicitly separate [firmer-stop physical pilot](WING_LIMITS.md)
before spending more learning time: preserve ranges, masses, actuation and all
other physics, strengthen only the wing limit solver response, and first verify
the inherited expert. This is an unvalidated parameter experiment, not an assumed
fix or a relaxation of flight acceptance. The original body remains the default.

### Wing-stop outcome and aerodynamic scope — 2026-09-13 02:28:17 UTC

The opt-in firmer-stop diagnostic from source `48008f1` preserves inherited
expert hover (1 s, 0.174 mm root RMSE) and forward flight (0.3 s, 0.102 mm RMSE).
The same readout01 student fails both tasks with original and firm stops.
Worst angular overshoot falls about 83%, but neither stability nor tracking is
rescued. All six runs are finite and warning-free; every state/model hash and
causal previous-action sequence is verified. No training took place in this
diagnostic. Keep the original default and both failure pairs.
[Detailed outcomes](runs/wing_stops_01/SUMMARY.md).

The user confirmed that the inherited, established aerodynamic approximation
is appropriate. Physical wing joints and bounded torques are integrated by
MuJoCo; the ellipsoid model estimates aerodynamic forces without a resolved
Navier–Stokes field or explicit wakes. This agreement does not change the
teacher-free student acceptance criteria. Flight remains unsolved; the next
controller work must address autonomous feedback stability rather than treat
offline decoder error or constrained wing travel as a success.

### Declared first-wingbeat calibration pilot

The readout01 cache exposes substantially larger wing errors during the first
5 ms than after 30 ms. On the two excluded expert episodes, first-millisecond
pitch MSE is 0.194/0.207 versus 0.0115/0.0121 after 30 ms. Actual unassisted
trajectories already diverge strongly within the first wingbeat. These observations
motivate startup weighting; they do not prove that it will establish flight.

Run one **10-second**, seed **54001**, learning-rate **0.001** calibration from
readout01 using its existing original frozen-feature corpus. Draw **40%** of each
1,024-example batch from the first **25 frames / 5 ms** of training episodes;
draw the rest uniformly from the full training histories. Divide per-axis errors
by training-target standard deviation (floor 0.1), giving small roll commands
attention relative to their scale. Held-out episodes remain excluded from all
normalization and sampling. Record startup and later validation errors separately.
Only the existing 1,542 wing-output parameters may change. No runtime startup
sequence, clock input, neural bypass, teacher or physics change is added.

Evaluate the same original-model hover/forward initial captures, then the fixed
walking cases and continuous transition; preserve all outcomes. Do not infer
physical flight from either improved startup error or fitted expert histories.

### Startup-weighting outcome and declared phase-coverage collection

Startup01 changes only the six existing wing outputs and still fails both
unassisted 0.3-second flights. Its startup pitch errors decrease, while roll
errors and overall excluded-episode MSE worsen (0.006163→0.007537). The fit took
10.000220 s and 16,875 updates. Do not promote it. Its report's generic
`corpus_sampling` sentence still says uniform, but the explicit startup sampling
fields correctly describe the actual 40% startup / 60% uniform implementation;
the generic sentence is corrected in subsequent reports without rewriting evidence.

The original eight demonstrations provide only **six training reset phases**;
held-out episodes use other phases. Thousands of later frames do not supply
additional zero-memory starts. Collect **64 episodes × 20 ms** using the unchanged
expert and original physics, seed **55001**. Random initial wing phase is a reset
condition only, never a student observation. Save every failure and causal action.
Replay the frozen readout01 graph on the valid new episodes, using the existing
whole-episode split with seed 1193, independently of the original split.

Then fit the same 1,542 wing parameters from readout01 for **10 seconds**, seed
**56001**, learning rate **0.001**, equal original/new corpus sampling, **40%**
startup draws from first 25 frames and training-only axis scaling. The graph,
upstream actor, utility, remaining motor outputs, physics and acceptance gates
stay fixed. Evaluate the original declared hover/forward captures; more diverse
assisted starts are data, not student flight success.

### Phase-coverage outcome and declared neural-memory handover diagnostic

All **64** short expert starts pass the original demonstration envelope:
6,400 physical frames / 1.28 simulated seconds, **4.150386 s** setup and
**10.873399 s** collection. Frozen replay of 64 sequences takes **4.049921 s**
total. Startup02 fits for **10.000448 s**, 16,517 updates, 13,800 distinct training
frames across original and new corpora. New-corpus excluded-episode wing MSE
improves **0.034882→0.010548**. Original excluded startup pitch MSE improves to
0.02762/0.02926, but both unassisted physical flights still fall. Keep this as
coverage evidence; do not promote the checkpoint.

Next test **readout01** in a paired diagnostic, not a new acceptance condition.
Four 150 ms physical episodes (speeds 0/5/10/20 cm/s), seed **57001**, begin with
**50 ms** of real expert actuation while the graph actor continuously observes
the causal sensor stream. At handover, execute **100% student** commands for the
remaining 100 ms. Compare one run retaining neural memory with another clearing
only neural memory at that same handover; match the entire physical prefix.
The teacher continues computing counterfactual labels for capture but cannot
contribute to executed commands after handover. Save per-frame executed fractions,
release neural state, actual actions and post-release metrics. Neither condition
can count as teacher-free flight acceptance. This isolates whether prior neural
history materially helps, before spending more training time on startup alone.

### Handover outcome and next motor-decoder scope

All four matched physical prefixes are bit-identical through the release frame.
After release, captured actions equal student outputs exactly; teacher labels
have zero actuator contribution. Retained-memory flights fail the envelope after
4.8/12.4/17.6/17.6 ms; cleared-memory counterparts fail after
6.8/12.2/12.8/11.4 ms. All eight complete their 100 ms post-release capture with
finite physics and zero warnings. Keeping prior neural history does not rescue
flight. See [the paired diagnostic](runs/flight_handover_01/SUMMARY.md).

The next learning pilot should train the **existing nonlinear motor decoder**
from cached actual motor-cell states, keeping the same deployed architecture,
encoder, measured graph and utility path. Include ground histories and preserve
the parent's non-wing outputs through explicit retention loss. Unlike the recent
six-row fits, changing shared motor hidden weights can affect legs; fixed ground
cases and continuous transitions are required again before any promotion. First
verify causal feature capture, whole-episode exclusion and unchanged upstream
parameters, then use a bounded GPU fit and teacher-free physical flight. This is
a declared next scope, not completed training or a claim that it will solve flight.

### Declared nonlinear motor-decoder pilot

Implement a causal frozen-actor feature cache for the **actual 815 motor-cell
states**. Replay every frame of each accepted episode, preserving full variable
lengths and independent recurrent columns; padded tails never become examples.
Capture parent outputs and external labels separately. The decoder never receives
raw observations or labels as inputs. Verify source/capture hashes and exclude
whole episodes with the existing split seed 1193.

Use readout01 as the parent and three corpora: original flight demonstrations,
64 short phase-diverse starts, and all accepted `retention_online01_01` ground
histories, including both complete stop/resume sequences. Ground loss preserves
the parent's full outputs on those histories. Flight loss supervises six wing
outputs with training-only axis scaling and retains the parent's other 72 outputs.
Sample ground with probability **0.5**, otherwise choose a flight corpus uniformly;
**25%** of flight draws come from each episode's first 25 frames / 5 ms.

Fit only the **existing nonlinear motor decoder**, 815→256→78 with the existing
LayerNorm/tanh layers: **230,572 parameters**, zero new deployed parameters.
The encoder, intrinsic core, utility/intention interfaces and measured graph stay
bitwise unchanged. Run **60 seconds** of RTX 4090 optimization, learning rate
**1e-4**, batch **1,024**, retention multiplier **4**, seed **58001**. This is
supervised decoder calibration on cached real graph states, not fresh RL or new
physical experience. Unlike wing-row-only fits, leg outputs can change; evaluate
both original unassisted flights, all six fixed ground cases and continuous
walk/stop/resume before any promotion. A smaller imitation error remains
insufficient evidence of flight or preserved walking.

### Nonlinear decoder outcome and declared physical-reward continuation

The motor caches preserve **12,000 ground frames**, including both 3,000-frame
stop/resume histories, **12,000 original flight frames** and **6,400 startup
frames**. Standalone decoder reconstruction matches parent outputs within
7.12e-6 maximum absolute action difference (declared tolerance 2e-5).
The **60.000718 s** fit makes 49,201 updates from 21,800 distinct training frames.
Excluded original/startup wing MSE improves to **0.002702/0.004972**, but both
unassisted flights fail. Five fixed ground cases stay stable; slow walking falls.
Continuous walk/stop/resume has permitted stable support, but stopping misses its
gate. Preserve the candidate without promotion.

Use this stronger imitation fit for one **60-second physical-reward PPO pilot**,
seed **59001**. Keep **32 native CPU worlds / 16 threads**, 20 kHz physics / 5 kHz
control, horizon 128, recurrent chunks 32, two epochs and 60 ms training episodes.
Use the new 64-start corpus for training-split airborne resets (48 starts), not
held-out starts. Keep the existing physical reward unchanged. Set exploration
standard deviation **0.01** (from 0.04), actor learning rate **1e-6** (from 5e-6),
and explicit ground rehearsal multiplier **4** (from 1) for this conservative
continuation. Fresh PPO/critic Adam is required after the supervised parent.
All actor interfaces and intrinsic cell parameters can learn; signed measured
connectivity stays fixed. No runtime teacher, wing oscillator, root force or
forced utility action is introduced. This is not a matched speed benchmark or
an attribution of outcomes to any one of these changes. Save all training falls,
measure physical collection and optimization separately, then evaluate the same
unassisted flights and complete ground suite before promotion.

## Wing-motion flight: approved change of physical task

Implementation started 2026-09-13 03:45:16 UTC after discussion. The user
explicitly prioritizes the integrated brain, retains full wing joint commands
and wing feedback, and approves replacing aerodynamic/inertial wing coupling
with flight forces derived from measured wing motion. This supersedes the
earlier no-applied-body-force condition for the new `wing_motion` preset only.
See [mechanism and parameters](WING_MOTION.md).

Before this change, nonlinear decoder01 fit for 60.000718 s but both flight
cases failed. Its PPO02 continuation took 60.647209 s; all 750 completed
episodes failed, mostly excessive tilt. Both are preserved with physical
evaluations, hashes and checkpoints. No promotion or claim of flight progress.

The first custom-force reference trial pitched over because the resultant
force was applied at thorax COM. Applying at whole-fly COM, with its lever
moment expressed at the thorax, yields a two-second reference hover: 0.733 mm
root RMSE, minimum upright 0.99948, minimum height 17.119 mm from a 20 mm
start, zero warnings. This is a training-only reference, not a learned brain.
The correction is part of force-model implementation, not a reward change.

First declared student pilot: evaluate preserved readout01 on this body, then
collect eight predetermined two-second reference episodes (seed61001, phase
and initial-heading variations, speeds 0/0/1/2 cm/s). Replay the full frozen
graph to cache real motor-cell states; reuse the matching ground cache. Fit
the existing 230,572-parameter nonlinear decoder for 60 seconds, seed62001,
Adam1e-4, ground probability0.5, retention4 and startup fraction0.25 over25
frames. Upstream actor and measured graph stay fixed during this warm start;
no new deployed network or runtime oscillator. Run independent student hover
and forward tests, then ground checks if there is flight traction. Preserve
all outcomes and classify training/reference performance separately.

### First wing-motion decoder outcome and next declared pilot

Source39960b2: all eight reference episodes pass. Parent readout01 and the
60-second decoder-only student both fail independent two-second hover.
Excluded flight wing MSE improves0.366067→0.000974, with startup0.446364→
0.005563, without autonomous control. Preserve those results.

The current decoder-only objective retains parent non-wing outputs even in
flight, while the reference flies with folded legs. The first student steps
therefore depart from the demonstrated body posture and rapidly change wing
outputs. Also, the phase-diverse references start with identical stationary
wing states but different unobserved teacher phases, yielding conflicting
cold-start labels. These are curriculum issues to address before more time.

Next declare eight two-second reference episodes with a consistent phase-zero
cold start (seed64001); phase is not supplied to the learner. Keep heading,
height and speed variations. Resume decoder01 and train the entire existing
actor for180 seconds, seed63001,16 recurrent sequences,32 learned steps plus
16 burn-in,25% reset starts,Adam1e-5,wing loss2 and ground retention4. Flight
labels cover all78 outputs, including flight posture; ground examples retain
walking. Intrinsic cell/interface parameters can learn; measured graph edges
stay fixed. Physical flight model and gates stay unchanged.

Flight and walking now share a500 Hz clock, so task weights must depend on
explicit per-example flight/ground identity, not clock speed. The training
loss now supports this and has a direct gradient/weight test; legacy mixed
clock tests pass. Run independent hover/forward and both original/new-preset
ground evaluations after training. Do not promote solely from imitation loss.

### Complete-actor imitation outcome and physical-reward pilot

Source d4ac5e7,seed63001:180.274561 s of training,393 updates,14,000 unique
training frames. Validation motor MSE0.103644→0.020944. Both two-second
student flights fail. Five of six original-model ground cases remain stable
(slow walking falls); only the left-turn case is stable in the new physical
preset. All raw ground gates fail. Preserve all results and the earlier
accepted walking reference; no promotion. Changed observation/body dynamics
require their own ground training before unified deployment.

Next declared pilot: physical-outcome PPO on `wing_motion`, resume brain01,
32 native CPU worlds,16 threads,64-step rollouts,16-step recurrent chunks,
two epochs,0.5 s episodes,180 s wall cap,seed65001,Adam3e-6,noise0.03,
explicit ground rehearsal weight4. Use only first training-split frames from
phase-zero corpus02 for resets; no teacher acts in live rollouts. Physics,
force law and the existing airborne reward stay fixed; rate-based reward
uses the actual2 ms interval. This is the first physical-reward learning
trial for this force model, separate from the two imitation stages.

The batch uses the same force implementation as native evaluation. PPO
failure traces now retain causal wing activity and applied wrench. Reset
validation rejects a corpus from another flight model, even if array shapes
match. Evaluation still runs the student without a teacher.

### Physical reward outcome and action-selection diagnostic

PPO01 runs182.873925 s,129,024 physical transitions /258.048 aggregate sim
seconds,211 PPO updates. Of716 completed episodes,127 reach the0.5 s timeout
without leaving the training height/orientation envelope;589 fail. Timeouts
are not tracking success. Quarterly survival counts37/31/21/38 out of179
episodes each show no sustained improvement trend. Both independent mean-
action two-second tests fail. Do not promote or blindly extend this run.

The imitation parent first crosses below8 mm at0.490 s; the PPO checkpoint
does so at0.244 s in the declared hover case. Thus imitation did produce
transient airborne control despite failing the full gate; PPO did not extend
it. Its motor standard deviation stays about0.03. Test the actual learned
stochastic PPO policy separately at predetermined seeds66001–66004, all
two-second hover trials, without more training. Preserve mean-action tests
and every sampled-policy outcome. Categorical activity plus the checkpoint's
motor distribution matches training action selection; it is not a new controller.


All four sampled-policy hover trials failed. This rules out that particular mean-versus-sampled action mismatch as a rescue for PPO01. Full failure clips, report/state/model hashes and three diagnostic checkpoints are retained. No candidate is promoted.

### Next declared feedback and learning-window repair — 2026-09-13 04:36:34 UTC

The prior goal turn made evidence-backed progress: the wing-driven force law works under a reference, student imitation develops transient lift, and PPO fails to extend it. No job remains live. The next change addresses observed curriculum limitations instead of extending PPO unchanged.

The reference uses altitude while the student has no height measurement. Add declared current height and requested height through the existing sensory encoder, with neutral migration and causal/native-batch checks. This is ideal simulator sensing, not learned camera perception. Keep the measured graph, motor routing, force law and all 78 outputs intact. Existing checkpoints and observation schemas remain loadable.

Increase supervised recurrent context to span several 12 Hz wing strokes; the prior 64 ms supervised window was shorter than a full 83 ms stroke. This is a testable hypothesis, not a proven cause. Use a bounded pilot, retain ground examples and independently evaluate actual unassisted flight before extending training. Then use corrective examples from student-visited states if the new candidate still drifts. No runtime teacher or wing oscillator is introduced.


The precise pilot resumes **wing_motion_brain_01**, adds two sensor columns (395 → 397 inputs; 256 added parameters), and uses eight parallel recurrent sequences, 128 supervised steps plus 64 burn-in at 500 Hz. This gives 256 ms of differentiable history (about three reference strokes), plus 128 ms of burn-in. The smaller sequence batch controls full-graph GPU memory while doubling examples per update versus the preceding 16 × 32 setting. It is explicitly a changed recipe, not a matched speed benchmark.

Use phase-zero reference corpus02 and unchanged online01 walking-retention data, 25% reset starts, Adam 1e-5, flight wing loss 2 and ground weight 4, seed 67001, **180-second training cap**. Fresh Adam is necessary for the expanded sensory matrix and is recorded. First verify full-graph neutral migration on CUDA. Then independently evaluate two-second hover/forward and retained ground commands. Do not extend solely on imitation loss. All timing, excluded episodes, input-command sources and failed gates remain explicit.


The first CUDA height migration check (`a87078e`) stopped before training. Maximum action differences were 1.40e-6 / 1.49e-6 at the two clocks, versus 1.74e-6 / 1.94e-6 when repeating the unchanged parent. Both exceed the inherited 1e-6 absolute action threshold. Preserve that failed report; no optimization ran.

For numerical compatibility only, retain the original fixed-tolerance result and additionally compare against twice the measured unchanged-parent variability, capped at absolute 1e-5 action / 1e-4 neural state / 1e-6 utility. This explicitly calibrated roundoff rule is unit-tested against larger mismatches and cannot change physical gates. Rerun under a new report name before the declared pilot; weights, data, task and optimizer recipe remain unchanged.

A read-only analysis of brain01's retained hover shows appreciable wing motion through the rise and fall: mean absolute left/right sweep speeds are about 45/29 rad/s at 0.15–0.30 s. Rest selection occupies 35.5% of that window, while the wings become unequal. Correlation does not identify the cause. If the height-context pilot still fails, use one paired `explore` utility intervention on the same checkpoint and initial state to test whether activity selection is disrupting motor flight. It is explicitly ineligible for policy acceptance and introduces no teacher or body force changes.

### Height outcome and motor-only course correction — 2026-09-13 05:04:49 UTC

Height01 trained for 181.722557 s (106 updates / 108,544 reused examples), reducing
validation motor MSE from 0.0207893 to 0.0179357. Both two-second flights failed,
as did all six original ground cases. The paired fixed-explore intervention also
failed hover (21.249 mm root RMSE versus 18.818 mm without intervention). No
candidate was promoted. Activity selection alone did not rescue this checkpoint.

The user requested motor learning first: stand idle, walk, hover; defer utility
and survival intentions. They also required the same physical body everywhere.
The new [motor pilot](MOTOR_FOCUS.md) uses the canonical wing-motion body for all
three tasks, with a checkpoint-enforced physics fingerprint. No aerodynamic
body or legacy walking preset is used in the new training/evaluation loop.

Before optimization, declare 32 worlds (11 stand / 11 walk / 10 hover), 180 s,
seed71001, fresh Adam1e-5, 32-step recurrent chunks and constant80% reference
assistance. Resume the preserved angle01 walking diagnostic, add neutral height
inputs, freeze utility selection/context weights and train current-state motor
corrections. This changes the recipe deliberately; it is not a speed benchmark.
Evaluation seed72001 uses the same saved actor across all tasks without assistance.
Keep all failed gates and training evidence. The full survival goal remains open.

Motor-focus01 (source90da9e0) completed180.624521 s,162updates,165,888 physical
transitions /331.776 aggregate sim seconds. All160 completed assisted episodes
reached two seconds without an envelope failure. All three unassisted cases
failed: stand/walk/hover first leave the envelope at0.418/0.268/0.058 s. Initial
hover sweep speeds are only4.78/10.91 rad/s averaged across100 ms, insufficient
for support. The complete three-case video retains all failures and was opened
for the user. Motor-only focus has not yet solved these primitives.

The first review mistakenly reported prohibited support from only the final
control interval. It still failed every case independently on posture. Preserve
that original report/video; fix the reduction to take the full-clip maximum,
add a regression check for an early transient contact and repeat evaluation
under a new name. Training, physics and acceptance thresholds are unchanged.

Next declared continuation: **motor_focus02**, resume01,180 s, seed71002,
32worlds/16threads/32-stepchunks, same Adam1e-5 and task loss, but reduce
reference assistance from80% to25%. Fresh optimizer is explicitly recorded.
This tests current-state corrections where the student has more control; do not
interpret assisted survival as autonomous acceptance. Evaluate all three tasks
teacher-free at the same development seed72001. No additional physics changes.

Motor-focus02 completed180.300127 s,155updates,158,720transitions. Of618 completed
assisted episodes,99 reach two seconds and519 fail (stand23/60falls,walk17/59,
hover479/499). Timeouts increase late in training, but all three independent
cases still fail. First envelope exits are0.388/0.478/0.064 s for stand/walk/hover.
Only the walking duration improves; this does not establish learned motor
reliability. Preserve the second full video and its actual neural activity.
Both new videos were opened automatically after full decode and visual review.

No further unchanged continuation is justified by these two pilots alone.
The next motor gate is stable unassisted standing on this same body, followed
by walking and hover with explicit retention of already-working primitives.
Keep a single command-conditioned checkpoint throughout; do not return to
the earlier physical preset or activate utility selection prematurely.

### Ground-control isolation — 2026-09-13 05:47:44 UTC

The previous goal turn made concrete progress: one canonical body, motor-only
training and two measured failed pilots, with complete reviewed videos and
verified causal traces. Before more training, test matched current-state
reference substitutions on the same motor-focus02 checkpoint and body. For both
stand and1 cm/s walk, compare pure student, reference wings, reference19
nonwalking channels, reference6 foot-adhesion channels, reference48 leg joints,
reference legs+adhesion, and full reference. Fixed initial headings−0.1/+0.1 rad,
one second per case,14 worlds, no learning or resets. Every substitution is a
diagnostic and is ineligible for policy acceptance. It should distinguish
unintended extra-body commands from inaccurate leg or adhesion control.

Interventions01 completed14 one-second cases with no numerical warnings.
Pure-student stand/walk leave the envelope at0.388/0.458 s. Replacing just
the six wing commands with the quiet reference keeps both upright, with no
prohibited support. Replacing foot adhesion or leg joints also stabilizes the
body, so this is not proof that wings are the sole source of error. Walking
speed remains poor under quiet wings (0.961 cm/s RMSE for a1 cm/s target).

Next bounded correction: **motor_focus03**, resume02,180 s,seed71003,32worlds,
16threads,32-stepchunks, fresh Adam1e-5 and25% reference assistance. Keep the
same physical model and all78 learned outputs. Add **2× wing-channel MSE on
ground tasks**, matching the existing hover wing weight. Previously ground
wing error was diluted among78 outputs; only hover received the extra term.
The runtime actor still controls its wings; no output override or force gate
is added. Evaluate all three tasks teacher-free at development seed72001.

Motor-focus03 completed181.184887 s /160updates /163,840transitions;152 of160
assisted episodes reach two seconds (47/55stand,55/55walk,50/50hover). All8 falls
were during standing training and retain verified causal traces. Both unassisted
two-second ground cases are now upright with valid foot support. Heading/speed
tracking gates still fail and unassisted hover still falls.

An explicitly extended native evaluation uses the same physical fingerprint,
seed73001, hold and1 cm/s walk for five seconds, no teacher or output mask.
Standing remains upright with zero prohibited support and0.801 mm drift.
Walking first leaves the envelope at3.28 s. Both raw tracking gates still fail.
Complete6-second three-task and10-second ground reviews were decoded, inspected
and opened automatically. Preserve03 as ground-stability progress, not a release.

The next hover diagnosis should inspect reference-target observability: the
current wing teacher reads a private12 Hz phase clock, whereas the actor receives
physical feedback and recurrent memory. A current-wing-state reference may make
the correction target easier to learn. This is a hypothesis, not an established
cause or permission to add a runtime wing generator. Keep the same body, full
actor and learned ground control while testing it before another training run.

### Initial-form correction — 2026-09-13 06:19:24 UTC

User review identifies body crouching during idle and wings dipping into the
body during both idle and walking. Uprightness alone missed this. Prioritize
this correction before the proposed hover phase-observability diagnosis.

A two-world, two-second bounded initial-position reference works with either
zero or half foot adhesion: no forbidden support, minimum uprightness above
0.99987, full-hinge RMS about0.0254 rad. Select **zero adhesion** for idle.
No physical parameter, contact, mass, joint limit or runtime action mask changes.
The first probe's terminal output was lost during context compaction; no process
remained. Its explicit repeat is saved as initial_pose_probe_02, never reported
as training.

Add opt-in `--ground-posture`: all position-actuated joints target their canonical
initial angles during stand. Passive hinges are measured too. Both stand and
walk receive wing restoring targets: torque=0.02×(initial-angle)−0.00015×velocity,
bounded at the unchanged0.03 torque limit. Walking leg labels and hover labels
remain unchanged. Zero torque alone did not actively restore a displaced wing.
Measure three joint groups independently, body-height loss and wing speed; do
not dilute six wings among102 hinges. Gate ground wing excursion at0.2 rad and
RMS speed at2 rad/s; idle additionally requires each group RMS below0.15 rad
and less than10% body-height loss. Keep old tracking gates.

This remains **online corrective imitation**, not PPO. The optimized loss fits
commands that restore posture; measured physical posture scores do not receive
gradients through CPU MuJoCo. The full102-hinge reference is saved in checkpoint
and evaluation reports. No deployed teacher or pose reset is added.

Reference01: stand and hover pass, walking preserves upright valid support but
still fails heading/speed gates. Both ground references hold wings exactly at
rest. Standing group angle RMS: legs0.01199, wings0, remaining body0.04343 rad;
zero height loss. Three worlds×2 s,4.118766 s capture,3.312887 s setup. This is
a reference demonstration only. Eight focused tests pass including displaced
wing recovery using bounded physical actuation and unchanged hover labels.

Declare motor_focus04: resume03, seed71004,180 seconds,32 worlds(11/11/10),
16 physics threads,32-step chunks,fresh Adam1e-5,25% teacher assistance. Activate
ground posture targets and increase ground wing imitation weight2→10. Everything
else, including the one actor and canonical body, stays the same. Evaluate all
three cases teacher-free for5 seconds at seed72001, retaining all failures.

Motor-focus04 completed180.182772s/161updates/164,864transitions. Despite
assisted improvement, all unassisted5-second cases fail. Stand/walk leave the
envelope at0.344/0.338s. The initial stand wing action reaches−0.155 normalized
when its correct rest target is0. At100ms all six stand wing commands push in
the direction of displacement instead of restoring it. Reference assistance
kept training states near rest; that accuracy did not transfer to autonomous
error states. Preserve the complete failed evaluation and film.

Declare motor_focus05: resume04,180s,seed71005,same32worlds/body/network/targets/
loss/Adam settings. Change only **teacher mixture25%→0%**: execute student
commands exclusively, while obtaining corrective imitation labels from the
actual states the student visits. This remains supervised online learning,
not PPO, and keeps one actor for all commands. Evaluate the same5-second cases
at development seed72001. Do not modify wing mass, contacts or runtime outputs.

Motor-focus05 executes only student actions during learning. After181.025272s,
153updates/156,672transitions, both ground cases remain upright with valid
support for5s. Standing height loss is3.48%;max wing deviation0.298rad. Walking
max wing deviation0.250rad andwing velocity RMS1.686rad/s. Both strict posture
and tracking gates still fail;hover falls. This is clear recovery from04,with
remaining posture error.

Declare motor_focus06: same recipe as05, resume05,seed71006,**120seconds**.
Keepzero execution assistance,32worlds,allthree tasks,samebrain/body/targets/
weights. The justified extra allowance targets the remaining posture error
after autonomous ground survival recovered. Evaluate allthree5-second cases
at seed72001;retain05 and open its review while06 trains.

Motor-focus06 preserves five-second ground stability. Stand height loss drops
3.48%→1.01%, leg RMS0.178→0.142rad and remaining-body RMS0.0767→0.0587rad.
Ground wing errors remain too large visually:stand/walk maxima0.270/0.270rad.
The wing velocity checks pass, but angle checks fail. Hover remains failed.

Declare motor_focus07: resume06,seed71007,120s,zero execution assistance,32worlds
and unchanged targets/body/actor. Increase only ground wing loss10→100 to give
rest-wing accuracy priority over small errors across the other72 outputs. This
is a bounded targeted accuracy check, not evidence the issue is already fixed.
Keep every prior checkpoint/video; evaluate allthree5-second cases at72001.

Motor-focus07 completed120.509577s/105updates/107,520transitions. Both ground
cases stay upright for5s,but wing RMS worsens to0.1443/0.0830rad(stand/walk)
from06's0.1140/0.0631. Stand sag improves slightly to0.751%. Keep06 as the
preferred ground checkpoint;neither passes full posture/tracking gates,and
hover remains failed. Do not escalate weights again without a new diagnosis.

Across04–07:one physical body/oneactor ancestry;four bounded pilots with all
outputs learned,all failures retained. Addedinitial-formtargets,direct wing
angle/velocity correction labels,and physical pose measurements. Body collapse
is substantially improved,but rest-wing accuracy still needs work. No claim
of recovered biological behavior or completedflight is supported.

### Wing-response audit — 2026-09-13 06:56:54 UTC

The previous goal turn is **progress**: committed initial-form learning, four
measured pilots and reviewed films, better five-second ground stability, and
evidence that a larger wing-loss weight does not fix rest-angle accuracy.
Mac and GPU are at8be7554; no training process remains live.

“Older walking inputs” refers only to the inherited383-feature observation
prefix. It already includes all hinge angles and velocities, including wings.
The additional continuous wing channels were introduced during aerodynamic
experiments to avoid saturation: six velocities/2000rad/s, six angles/pi, then
two altitude inputs/2cm. All current tasks still use the same non-aerodynamic
wing-motion physical body. No old physics model is being reintroduced.

Before another training change, declare a frozen neural-response diagnostic on
preferred06's actual saved histories. At0,0.1,0.5 and1s, perturb each wing angle
by±0.05rad or velocity by±2rad/s in both corresponding sensory paths. Compare
action responses after1,5 and25 held-input neural updates. Hold every other
input and copied prior neural state fixed. This is a counterfactual sensory
probe with zero physical transitions and zero learning; it is not an autonomous
rollout or biological validation. Check original-history action replay and
retain exact source/capture hashes. This should distinguish wrong-sign response,
weak response and simple constant bias before changing the training recipe.

The response audit completes in3.985166s plus2.563717s setup,75 parallel neural
sequences,zero physical transitions/learning. Original-history actions match
the saved capture within5.67e-7. Ground angle/velocity responses are weak and
sometimes have the wrong sign; normalization is not proven to be the cause.

Implement opt-in wing-response supervision: for eight ground worlds per action,
copy the actual prior neural state and make a positive/negative perturbation of
one sampled wing angle(±0.05rad) or speed(±2rad/s). Recompute both corresponding
sensory paths. Compare the resulting six-wing action difference to the bounded
reference difference, normalized by the requested unsaturated response. This
trains local response and cross-axis independence, supplementing ordinary motor
labels; it adds no runtime controller, physics worlds or new learned module.
Synthetic examples are explicitly excluded from physical transition counts.

Declare motor_focus08: preferred06 parent,seed71008,180s,32worlds,16 CPU threads,
32-step chunks,fresh Adam1e-5,zero execution assistance,ground posture active,
ground wing weight10,wing-response weight1,eight sampled ground worlds. Inputs,
body mechanics and78-output graph actor remain unchanged. Evaluate allthree
five-second cases at72001;preserve every failed case.

Motor-focus08 finishes181.009871s/127updates/130,048physicaltransitions.
Both ground cases remain upright with permitted support for5s, but wing RMS
worsens to0.1463/0.0898rad and standing sag to11.27%. Hover still falls.
All full gates fail; no numerical warnings. Keep06 preferred. All245failure
traces and three complete captures pass hash/causality/bounded-action checks.

Response02 replays08's saved actions within4.62e-7. Mean ground restoring
angle/velocity gains are-0.03508/-0.001147, with6/48 wrong-sign responses each.
Those are closer to the targets than response01's aggregates, but the probes
use each checkpoint's own histories. They are descriptive, not matched causal
estimates. Better local response does not imply better physical control:
absolute bias, cross-axis coupling and the weak target gain remain unresolved.
Do not promote08 or simply run it longer based on the auxiliary loss.

### Frozen wing-output fit — 2026-09-13 07:26:59 UTC

Previous turn is progress: new response supervision and an evaluated failed
pilot narrow the problem;06 remains preferred. Mac/GPU synchronized atb3a9891;
GPU is idle before this step. No external blocker.

Declare ground_readout01 using parent06:32worlds,16 CPU threads,2s perworld,
seed73009. Execute only parent actions, no resets, retain every failed history.
Every10actions copy actual prior neural state and probe each ground wing angle
by±0.2rad and speed by±5rad/s. Capture the existing256-unit motor hidden layer;
the only labels are bounded ground resting-wing corrections and inherited hover
commands. The synthetic probes add zero physical transitions. Hold out the last
world of each task, including all its sensory variants.

Fit only the existing6×257 wing output weights/biases by residual ridge
regression in inverse-tanh action space. Equal task weights; within each ground
task, nominal and synthetic examples each receive half its weight. Select
alpha from[1,0.1,0.01,0.001,0.0001] by whole-world validation action error.
No new deployed module, input normalization, physics changes or action mask.
Keep all upstream parameters/non-wing rows exactly fixed; preserve motor-only
context and physical fingerprint in the checkpoint, omit stale Adam state.

Evaluate the fitted checkpoint on allthree full5s cases at developmentseed72001,
with no teacher and no resets. A better offline fit does not establish physical
success. This diagnostic fit tests whether the existing motor features can
decode the required wing feedback before spending another end-to-end pilot.

Readout01 collects32,000physicaltransitions in31.150358s, then fits in0.170996s.
Selectedalpha0.001 lowers held-out command MSE72.9%, but physical standing falls,
walking wing RMS worsens0.0631→0.2298rad and hover still fails. Preserve allresults;
no promotion. Parent06's standing inputs show zero clipping in all ground wing
angle/speed/previous-command channels, including both sensory paths.

Declare previous-command response probes on06 and readout01, each with its own
captured history and exact checkpoint verification. Perturb each previous wing
command by±0.02, preserving every joint measurement and other input. Compare
response after1/5/25 held-input neural updates. Frame0 has the same initial
physical inputs and zero memory across candidates. Later histories differ.
Report full6×6 response and spectral radius; this is a local sensory diagnostic,
not a complete closed-loop stability proof. No physics or learning in the probe.

Command probes complete in3.530120/3.515242s, setup3.776308/3.597831s.
Ground local command-response spectral radii remain below0.26 for both actors;
the sampled maps do not show strong self-amplification. No command-invariance
penalty is justified from this evidence alone. Frozen single-update wing-plant
linearization at the initial collected states also does not explain the later
readout failure; do not present a local approximation as full-loop stability.

Declare readout02 as one dataset-aggregation follow-up. Collect32worlds×2s
with readout01 acting,seed73010,all previous collection settings unchanged.
Pool with the original06 corpus after proving every upstream/hidden feature
parameter, physical fingerprint, graph and observation schema identical.
Keep complete-world validation partitions; preserve every source/capture hash.
Fit the same six output rows with the same ridge grid/weighting, then evaluate
allthree5s cases at72001. This tests distribution shift using the candidate's
actual trajectories, not an unchanged continuation or an additional runtime
controller. Keep06 selected until physical evidence supports a better result.

Readout02 collects another32,000physicaltransitions in31.473841s, pools112,000
examples in9.079576s and fits in0.212640s. Alpha0.1 is selected. Standing remains
upright but its wing RMS is0.4342rad;walking falls;hover loses altitude. All
complete gates fail. New-corpus validation MSE0.05639→0.02840 is not comparable
to01's different validation corpus. Keep06 preferred. Stop six-output-row
calibration: it improves static label fit without producing a better controller.

Next motor work should update the sensory-to-motor representation from actual
ground trajectories, with initial-form and wing feedback supervision, before
resuming the harder flight curriculum. Preserve one canonical body and one
command-conditioned actor. No new actuation model, input bypass or runtime
wing override has been introduced. Full survival integration remains deferred
by the user's motor-first instruction.

### Ground motor curriculum — 2026-09-13 08:04:36 UTC

Previous turn is progress: two decoder-only candidates and command-response
probes were implemented, measured, archived and reviewed. They did not improve
physical control, so06 remains preferred. Mac/GPU synchronized at12186ff;
no GPU training process is live at this continuation start.

Declare motor_focus09: resume preferred06, full sensory encoder/intrinsic cell
dynamics/motor decoder trainable, fixed graph and frozen inactive utility heads.
32worlds are16standing/16walking for this curriculum stage. Hover is not trained
in this stage; it remains a required task and is still evaluated with the same
checkpoint. This stages motor learning, not a separate deployed policy.

Use180s,freshAdam1e-5,32actionchunks,16CPUphysics threads,teacher execution
mixture0,ground-posturetargets,ground-wingweight10,paired-responseweight1 with
eight ground worlds. At training resets only, perturb wing angles by±0.15rad
(clipped to physical joint limits) and speeds by±2rad/s using a separate RNG.
The nominal full initial-pose target is unchanged. Other physical reset state
and all body/force parameters remain unchanged. Trainingseed71009,2sepisodes;
evaluation uses allthree complete5s cases atdevelopmentseed72001 with no
disturbance,teacher or reset. Save failed cases and open the complete film.

Why this change: cached output fits could not deliver stable wing feedback.
The new stage trains the complete existing representation on actual ground
disturbances, without simultaneous hover imitation gradients. It is corrective
online imitation, not PPO. No runtime wing mask, controller bypass, actuator
change or observation-normalization change is introduced.

Motor-focus09 completes181.150672s/129updates/132,096physicaltransitions.
All128 completed ground episodes reach the2s timeout, without assistance.
Five-second evaluation keeps both ground cases upright with permitted support,
but standing/walking wing RMS0.1637/0.0751rad does not improve preferred06.
Standing height loss is0.508%;full tracking/posture gates still fail. Hover,
which was not trained in this stage, still falls. Do not promote09.

Declare motor_focus10 as one bounded optimizer-rate continuation from09:
same ground curriculum,32worlds(16/16),reset disturbances,targets,response loss,
zero execution assistance and180s budget; freshAdam1e-4 instead of1e-5.
Seed71010;evaluate allthree full5s cases at72001. Test whether the conservative
step size limits representation learning; do not assume it is the sole cause.
This changes the optimization rate only, with one continuing actor and identical
body/force parameters. No new physics, controller, observation or reward recipe.

### Physical motor rewards — 2026-09-13 08:34:30 UTC

The last user-question turn clarified inherited input encoding; it made no
implementation progress. Revalidated the terminal09/10 training and transfer
handles and existing artifacts before continuing. No job was restarted.
Motor-focus10 completed180.398s/132,096transitions,128episodes/5failures.
Standing is quieter but wings remain inaccurate and walking becomes nearly
stationary. Allthree full task gates fail. Preserve preferred06. Both09/10
complete three-task films are now decoded, inspected and opened for review.

Corrective imitation has improved support but has not produced resting wings
and good command tracking together. Implement an explicit motor-only path in
the existing recurrent PPO trainer: utility/intention heads stay frozen, all78
motor channels remain learned, same397inputs/4graphupdates and exact physical
fingerprint. A separate training-only critic estimates return. No teacher,
rehearsal data, output mask, pose writes or new deployed controller is used.

Rewards are calculated from each world's post-action physical state every2ms.
Retain velocity/yaw tracking, uprightness, valid support and action smoothness;
add resting-wing position/stillness rewards in both ground tasks, initial leg
pose only when standing, and initial body pose/height on the ground. Pose terms
use weight/(1+MSE/scale²), avoiding a fully saturated exponential at large error.
Rates are multiplied by0.002s; physical failure has one penalty before reset.
Timeouts bootstrap the final physical state; recurrent memory is cleared only
for reset worlds. Utility log probabilities are excluded from motor-only PPO.

Declare motor_ppo01: resume preferred motor_focus06,32worlds(16stand/16walk),
16CPUphysics threads,180s budget,128-actionrollouts,16-actionrecurrentchunks,
2epochs,Adam1e-5,initial latent action noise0.02,targetKL0.03,entropy0.001,
gamma0.998 andlambda0.99 at500Hz. The longer discount/advantage horizon gives
wing position and balance consequences time to influence credit assignment.
Two-second episodes;training resets perturb wing angles±0.15rad andspeed±2rad/s.
Seed81001. Ground-only curriculum stages learning; hover remains required and
is evaluated from the same resulting checkpoint. Evaluate allthree complete5s
cases,seed72001, no perturbation/teacher/reset; open the complete film. Compare
physical results with06, not just training reward. This is a changed learning
method, not a matched speed benchmark or a new deployed policy.

Pre-run validation:89tests pass in56.26s (15known upstreamwarnings), including
canonical wing decoupling and both old utility-PPO and new motor-only likelihood
paths. Ruff and diff whitespace checks pass. No physics or deployed architecture
change is included. GPU is idle with about20GiB free before launching the pilot.

Launch naming correction: `motor_ppo_01` was already used by the September12
walking/utility PPO experiment (source082bdc2). The CLI's exclusive directory
creation refused the new run before model setup or training. Its saved checkpoint,
report, trajectories and archived evidence remain intact; the scratch launch log
was reused. The newly declared canonical-body motor-only trial is named
**ground_outcome_01**, with exactly the parameters above. No previous experiment
is resumed or overwritten by this naming correction.

Ground-outcome01 completes182.957318s,47rollouts/47PPOupdates and192,512physical
transitions. Physical reward gradients reach the intrinsic dynamics;utility and
intention weights remain unchanged. Five-second evaluation:standing falls,
walking remains upright but misses speed/yaw/wing gates,hover falls. Zero MuJoCo
warnings. Do not promote. KL early stopping allowed only one update each rollout;
observed post-update KL frequently exceeds0.03, so the1e-5 step size is too large
for this narrow78-dimensional exploration distribution. This is evidence for
reducing the optimization step before changing the reward or body.

Declare **ground_outcome_02**: restart from preferred06 with fresh PPO/critic,
all01 settings unchanged except actor Adam learning rate1e-6 (ten times smaller)
and trainingseed81002. Still180s,32worlds,16stand/16walk,128rollout/16recurrent
chunks,2epochs,noise0.02,gamma0.998/lambda0.99,ground reset disturbances and
2sepisodes. Same actor/physical body/reward;evaluate allthree5s cases at72001 and
open the complete video. No claim of improvement until that physical review.

### Ground-outcome result — 2026-09-13 08:55:59 UTC

Ground-outcome02 completes182.308462s,32rollouts/288PPOupdates and131,072physical
transitions. The smaller learning rate resolves the one-update bottleneck;
128episodes include3physical failures. Both five-second ground cases are upright
with permitted supports. Standing wing RMS0.1486rad/max0.5787rad and6.76% height
loss are worse than preferred06. Walking wing RMS0.05966rad/max0.25262rad is
slightly better than06, but the complete tracking/posture gates still fail.
Hover falls. Zero numerical warnings;utility/intention weights stay unchanged.
Do not promote either ground-outcome checkpoint.

Both complete15s/750frame/50fps/1x films are decoded, visually inspected and
opened automatically. Captures, models, checkpoints and every failure trace are
hash-verified; previous-action feedback is causal and controls are bounded.
The paired comparison verifies identical initial qpos, qvel, observation and
commands to06 in every case. These are reused development cases, not held-out
robustness evidence. Source/reward/optimizer configurations and exact timings
are archived in each run directory.

Recorded-state windows add useful diagnosis:01 standing first leaves the
physical envelope at3.616s, beyond its2s training episodes.02 fixes that fall,
but its standing wing error persists over2–5s (RMS0.1469rad,max0.3622rad).
Walking last-three-second wing max0.2060rad approaches the0.2rad gate, but full
history still fails. Neither startup trimming nor a longer identical run is
justified as a solution to persistent standing wing bias.

This goal turn is progress: canonical motor-only physical PPO is implemented
and tested, two bounded trials are complete, and their failures change the
next action. Preferred06 remains intact. Next work should address sustained
rest-wing feedback while preserving ground movement, rather than repeatedly
increasing PPO runtime. Existing corrective wing supervision can be evaluated
as a training-only auxiliary to physical rewards; no runtime helper or new
brain is authorized or needed. Declare any such changed objective before its
pilot. Hover, motor transitions and later needs/utility/survival integration
remain required and incomplete; the full goal stays active.

### Wing correction within physical PPO — 2026-09-13 08:59:12 UTC

Previous goal turn is progress: implemented canonical motor-only PPO, completed
and reviewed two bounded trials, verified their physical comparison, and kept
preferred06 after both failed the complete task. Mac and GPU are synchronized at
fd5eeeb; authoritative process check shows no live embodied-fly training job.

The smaller PPO step fixes the update bottleneck but leaves sustained standing
wing bias. Add an opt-in training-only corrective wing loss to PPO. Labels use
current pre-action measured wing angles/speeds and the existing initial-pose
reference: clip((0.02*(q_rest-q)-0.00015*qvel)/0.03,-1,1). This reuses the
established correction without changing the force law, initial body, joints,
limits, actor, input scaling or deployed control. Only the six wing commands
receive supervised labels; body and walking-leg actions are learned from the
physical reward. All78 channels still execute the sampled actor distribution.
No reference commands are mixed into physical control, and no legacy walking
corpus or utility loss is introduced.

PPO and wing-supervision gradients are audited separately before their sum.
The report must not attribute supervised gradients to physical return. Count
replayed wing-label presentations separately from new physical transitions;
auxiliary compute is included in measured PPO optimization time. Failure
traces retain pre-action labels alongside the actually executed actions.
Motor checkpoints now enable posture gates by default in their saved config.

Declare ground_outcome03: restart preferred06, fresh actor/critic Adam,180s,
32worlds(16stand/16walk),16CPUthreads,128-actionrollouts,16-actionchunks,2epochs,
actorlr1e-6,noise0.02,targetKL0.03,entropy0.001,gamma0.998/lambda0.99. Keep2s
training episodes and the same wing reset disturbances (±0.15rad,±2rad/s).
Add100 times mean squared error over the six normalized wing commands to the
PPO objective. Seed81003. This is PPO with corrective wing supervision, not
pure physical-reward learning. Evaluate allthree complete5s cases at72001 from
one checkpoint, without teacher/reset/perturbation;retain and open the full
film. Compare against06 and pure-PPO02. Do not promote on supervised loss alone.

Focused checks:20tests pass in23.34s,11known warnings. These include physical
collection, preserved utility weights, both optimizer continuation paths,
separate reward/supervision gradient audits, exact shared-body fingerprint,
and restoring/braking wing-label direction without physical state writes.

Pre-run full validation:91tests pass in59.62s (15known upstreamwarnings),
including shared-body decoupling, both pure/hybrid motor-PPO paths and preserved
older controller tests. Ruff and whitespace checks pass. No physics or deployed
actor change is included in this pilot.

Ground-outcome03 completes182.342745s,131,072physicaltransitions and283PPOupdates.
Both ground cases remain upright on permitted support. Standing wing RMS falls
to0.05704rad from06's0.113997rad;walking falls to0.05276rad from0.063092rad.
Peak wing errors0.24198/0.24166rad still exceed0.2rad. Standing body height loss
3.479% is worse than06's1.008%, and heading/tracking gates remain failed. Hover
falls. No numerical warnings. This is useful ground-wing progress, not release.

The first gradient audit separates physical return (core biasL2:79.4092) from
weighted wing guidance (2.64913). Both reach the intrinsic dynamics; the report
does not attribute the latter to pure RL. Utility/intention weights remain fixed.

Declare ground_outcome04: continue03 for180s with identical reward, wing loss100,
actorlr1e-6,32worlds,rollout/chunk/epochs,noise,episode duration and reset ranges.
Retain03's PPO optimizer,critic and exploration state;seed81004. This allowance
is supported by measured wing improvement in both ground commands. Do not change
body, force law, targets or deployed architecture. Evaluate allthree complete5s
cases at72001, retain03, and open both complete films. No successful motor
release is claimed from the current development cases.

### Hybrid motor outcome and hover audit — 2026-09-13 09:22:48 UTC

Ground-outcome04 completes185.485408s/131,072transitions/298PPOupdates with
optimizer,critic and exploration state resumed from03. Both ground cases remain
upright with valid support; standing wing RMS0.06067rad/max0.31920rad and5.02%
height loss are worse than03. Walking wing RMS0.05306rad is similar but maximum
0.25946rad is worse. Hover falls and no full task gate passes. Do not extend this
same recipe again. Keep03 as the next motor-curriculum parent because it improves
wing posture in both ground commands; preserve06 as the previous comparison.
This is one development checkpoint for every command, not a per-command mixture
or an accepted motor release. Full-body gait/heading quality remains incomplete.

Both complete15s films are decoded (750frames each), visually inspected and
opened. Every capture/model/checkpoint and failure trace is verified. Wing-label
values are reconstructed from pre-action observations and physical actuator gains;
executed actions remain the sampled actor controls, not the labels. Paired
comparison confirms identical initial qpos,qvel,observation and commands across
06,pure-PPO02,hybrid03/04. Training seeds differ; this is development evidence,
not a generalization claim or a clean multi-seed causal estimate.

Hover-clock audit01 takes3.320554s setup/0.004622s execution on CPU, with zero
physics transitions and zero training. Holding all397 current sensor inputs and
physical state fixed, changing the reference timer changes sweep labels from
+0.52269 to-0.62776. The recurrent actor could encode phase in history; do not
claim the task is impossible or that this alone caused failed hover. The result
supports testing a teaching law based on current wing angle/speed and altitude
error, removing an extra timing burden while preserving the same actor/body.

Next available action: build and physically validate that state-based hover
reference on the canonical model, including startup from zero wing speed,
without modifying actuation, body masses or the wing-motion force law. A bounded
energy/phase response derived from measured wing coordinates is a candidate;
its gains and startup behavior must be tested, not assumed. It would generate
training labels only. After reference validation, declare a mixed ground/hover
curriculum from03 with the same shared actor. Preserve working ground behavior
and evaluate all commands from the resulting single checkpoint. No such new
flight reference or training has been implemented or claimed in this turn.

This turn is progress: wing-only guidance is implemented and tested, two GPU
trials and their full films are complete, resting-wing accuracy improves, and
an actual teacher-observability audit changes the next experiment. Hover,
transitions, utility/needs and final multi-agent survival integration remain
required; the full goal stays active.

### State-based hover reference and retained ground motor targets — September 13

The teacher timer audit led to a reference based on measured wing angle/speed,
body velocity and current/target altitude. It starts from still wings without a
clock. Five reference gain profiles preserve the same canonical body and force
law. The selected profile achieves about 1.21 mm full-window root error and
0.31 mm late vertical oscillation in three five-second airborne starts. This is
teacher performance, not learned flight. Initial source snapshots, all profiles
and failed soft-gain results are retained under runs/state_hover_gain_probe_01.

Added an optional frozen copy of the current motor actor as a training-only
reference for ground actions, preserving independent recurrent state and sharing
only immutable graph buffers. Ground wing targets still correct toward the
initial pose. Hover assistance and loss weighting are explicit per task. The
three-minute pilot recipe is declared under runs/state_hover_retention_01; it
has not yet run at this checkpoint. A single unassisted actor remains the
required evaluation path. The complete embodied suite passes 97 tests in
67.46 seconds, with 21 dependency warnings. Focused checks include unchanged
reference weights, per-world reset, no ground assistance, clock invariance,
physical startup and unchanged body fingerprints.

### State-hover retention01 result: walking wings improve; candidate rejected

The declared pilot completes180.791216s of learning,151,552physical transitions
and148updates on32worlds. Native CPU MuJoCo/mjbatch supplies physics; RTX4090
executes both the learning graph and frozen ground reference. Graph weights and
body fingerprint match the parent. The frozen reference, utility and intention
parameters stay unchanged; internal cell parameters receive nonzero gradients.

Unassisted5s evaluation: walking remains upright with permitted support, and its
wing-posture gate now passes (RMS0.0405rad,maximum0.1654rad). Standing fails after
0.542s and hover falls. These results reject promotion over ground_outcome03.
All initial physical states and397observations match the parent's review exactly.
The mixed training had44successful stand episodes,44successful walk episodes,
40assisted hover episodes and3walking failures; those counts do not establish
frozen-checkpoint success from a fresh neural state. Inspect the fresh-reset
response before blindly continuing training. Lower learning rate and startup
retention remain hypotheses. No additional pilot is running at this handoff.

All training/evaluation hashes,3failure traces, actual per-task action mixtures
and causal observation feedback are verified. The15s full diagnostic video is
fully decoded (750frames,1600×900,50fps), visually inspected and opened. The
separate5s reference hover clip is also decoded/inspected/opened and clearly
labeled training demonstration, with no MaleCNS actor. Its canonical-Linux
hover result matches the CPU/Mac reference; it does not establish learned hover.

This is progress on the active motor-learning priority. Same body, same single
command-conditioned actor, no restored aerodynamic preset, no runtime teacher.
The original walking sensor prefix carries current measurements; its historical
name refers to the input layout. Full motor acceptance and the broader learned
utility, flight transitions and multi-agent survival goal remain unfinished.

### Fresh-reset audit declaration

Previous goal turn made progress: a state-based hover teacher was validated,
new ground-retention training was implemented/tested, one complete GPU pilot
and its full failure-inclusive video were archived. The candidate is rejected;
its standing/hover failures determine the next action. No process was left
running and no external blocker is present.

Before more learning, motor_reset_audit compares the final checkpoint's training
and evaluation inference modes, exact initial observations before/after teacher
construction, and first actions for batch sizes3and32. It then physically rolls
out the same saved evaluation starts replicated into32worlds for one second,
using the frozen actor alone. This is a bounded diagnostic with zero learning;
no teacher action, physical parameter or acceptance threshold changes. It tests
setup/batch discrepancies and retains the current failure as evidence.

### Frozen-reset audit and smaller-step motor curriculum — September13

The reset audit reproduces the saved evaluation starts in32worlds. Inputs are
identical before/after training-reference construction; loaded/mode/batch first
actions differ by at most4.18e-7. All standing copies still fail at0.494s and all
hover copies at0.332s, while walking copies survive the one-second diagnostic.
These are duplicate starts, not independent trials. This rejects a changed input
schema or a failure restricted to the three-world evaluation batch. The original
wing-velocity channels still carry measured speed; extension scaling was inspected
and left unchanged. No physics or network changes were made in this audit.

Retention02 changes only learning rate1e-5→1e-6 from run01, with the same parent,
seed and all other settings. Both runs collect151,552transitions and148updates.
The smaller step preserves upright standing/walking, improves average resting-wing
error relative to parent03, but still fails peak-wing/command gates and hover.
Its181.192194s training and full three-command review are archived;02 becomes
the next development parent, with ground_outcome03 preserved.

Retention03 continues02 at1e-6 and removes hover collection assistance. The actor
executes every control; references remain labels only.180.599455s learning,
147,456transitions/144updates. Standing posture passes (wing maximum0.16559rad,
height loss1.366%), but walking fails after0.378s and hover still falls. Preserve
this diagnostic and keep02 as the better combined controller. All200saved failure
traces across02/03 and all model/checkpoint/capture hashes are verified.

The recorded02 flight trace already shows stroke reversals. Its first-quarter-
second mean modeled lift magnitude is0.934bodyweights versus0.991for the reference;
wing pitch is near nominal. This descriptive reconstruction excludes drag/body
orientation and uses different reference/actor target heights; it is not a matched
success comparison. A modest lift deficit can still produce substantial altitude
loss at this scale. We did not change the flight law to hide that deficit.

Both15s complete films (750frames each,1600×900/50fps/1x) are decoded, visually
inspected and opened. The final03result is not promoted. The next actionable
experiment is training only the existing six wing-output rows from02 while
freezing the upstream network and other output rows. This keeps one graph actor
and the same runtime shape. It needs parameter/inference invariance checks and
full physical evaluation, since changed wing feedback can still affect body
behavior. No such selective-output training is implemented or running yet.

This turn made progress: a concrete setup/batch audit ruled out a suspected
mismatch; a controlled learning-rate repeat restored ground behavior; an actor-
only continuation exposed continued cross-task interference while passing the
standing posture gate. The full motor/utility/survival goal remains active and
unfulfilled. There is no external blocker.

### Selective existing wing-output learning: implementation

Previous goal turn made progress: a matched reset audit, two complete GPU
pilots, full evaluations and two reviewed/opened films changed the next action.
Run02is the combined development parent;03passes standing posture but regresses
walking and still fails hover. No training process remained live at the start
of this turn and no external blocker is present.

Implemented an optional wing-output training subset in the existing motor
curriculum. Only the six final motor linear rows and biases may update (1,542
eligible parameters). All upstream state and72other output rows are checked
bitwise unchanged. Training hooks affect gradients only; checkpoints retain the
same397inputs/78outputs and ordinary runtime architecture. Existing full-actor
training remains the default. The frozen parent sees the same actual sensor
histories, allowing direct non-wing output comparisons throughout training.

The initial focused test failed in its fixture because generic deepcopy cannot
copy PyTorch sparse-CSR storage. Rebuilding the tiny actor and loading its state
fixes that fixture; the production frozen reference already shares immutable
graph buffers correctly. The second focused run passes12tests in19.21seconds.
It includes an actual training/save integration, changed wing weights, frozen
upstream/body rows, and identical-history non-wing inference. The full suite passes100tests in69.80seconds, with23dependency warnings.
The wing_output01pilot is declared but has not started. This stage is output-decoder imitation, with deliberately frozen
internal cell parameters, not a claim of end-to-end brain learning in this run.

### Selective wing-output trial outcome and next physical search — 2026-09-13 11:02 UTC

Wing_output01 completed:180.428s,174,080transitions,170updates. The selective parameter boundary is independently verified, but stand/walk wing RMS worsened and hover still fell. All495failure traces were verified. Its full15s film is decoded, inspected and opened. Retention02remains selected. The preceding goal turn made progress through implementation, a real GPU pilot, and full physical evidence; the intervening user question clarified sensor history without changing the controller.

A declared next experiment searches two constants in the existing sweep-output matrices, using physical outcomes instead of torque-label fit. Sixteen candidates run in two batches of32worlds: eight candidates, each with stand, walk and two airborne starts. Initial physical states are matched across all candidates. All actions still originate in the same frozen graph actor, with candidate-specific constant decoder rows. Selected constants are baked into ordinary actor weights; there is no new runtime module or sensor bypass. Ground retention constrains selection; a >=5% lower two-start hover cost is required. Short search results require separate full-length evaluation. This is parameter-space physical calibration, not PPO or full-core learning.

The new search passes3focused tests in3.78s and the complete103-test suite in71.78s (23dependency warnings). Ruff passes. Source, declared grid and archived preceding trial are committed before remote execution.

### Sweep search outcome and state-readout fit — 2026-09-13 11:14 UTC

The physical search completed70.556s after6.464s setup,64,000transitions/128aggregate simulated seconds. All16candidates and64case trajectories remain archived. No candidate met the declared combined improvement/ground-retention rule; exported weights exactly equal the parent. Independent checks verify exact initial observations across candidates, causal bounded actions, physical contract and selection. PeakCUDA811,394,048bytes.

The rejected gain1.3/bias0.02candidate has the best unconstrained hover cost. On its first search airborne start, height stays around1.9–2.5cm through1.5s, but forward drift grows; both sweep joints reach the1.5rad stop around1.75s and lose lift. Its ground wing posture is worse. A separately declared diagnostic export received the full5s/three-command evaluation, seed72003:standing/walking upright,hoverfailed,allstrictgatesfailed. It is not promoted. No physics or teacher assistance was changed.

The next executed implementation is canonical state-readout fitting. Actual frozen graph/decoder features from parent ground histories and successful state-hover reference history train the existing six wing rows by weighted ridge regression. Two focused tests pass in1.66s, including complete training/export and frozen non-wing/upstream state. This is offline readout learning, not PPO and not new physical experience. The fullsuite and run are pending at this checkpoint.

State-readout source passes the full105-test suite in72.93s with23known dependency warnings;Ruff passes. Allthree real fitting sources pass current-body, model/capture-hash and causal-feedback checks. The diagnostic film is decoded,inspected andopened. Commit this source before the boundedGPUfit.

### State-readout outcome and explicit boundary errors — 2026-09-13 11:23:36 UTC

State_readout01completed19.015356s total (4.521751setup,13.565234frozenGPUreplay,0.928369fit/packing/save). Five ridge candidates use6,000fitframes and1,500temporally-relatedvalidationframes fromthreealreadyrecordedhistories. Zero new physical experience. Selectedregularization0.01reduces reference-history wing error while preserving allupstream andnon-wingweights. Independent recomputation verifies the fit, validationmetrics andexactphysicalcontract.

Unassisted5s-per-command evaluation takes7.295372s setup +32.408363s capture, withzero numericalwarnings. Stand/walkstayuprightandproperlysupported butwingposturefails;hoverfalls. No candidate is promoted;retention02remainsselected. The full15s film is decoded,inspected andopened (750frames,1600×900,50fps,1x;60.334684srender/encode).

Recorded-state audit usesmj_forwardandtheexistingtraining-onlyreference,withoutintegration: at0.20s the actualsweepanglesareapproximately-1.45rad,actorcommands-0.32/-0.31pushoutward,whilereferencecommands+0.47/+0.48reverse. At0.002s actorstartupcommands0.22/0.27arebelowreference0.87/0.91. This is concrete evidence for training on actual startup and near-limit mistakes as well as successful-reference histories. It does not prove auniquephysical/neuralcause. Adding those actual-state corrections is the next action; it has not yet been implemented or executed.

This continuation made progress: archived the prior real trial, implemented andtested physical parametersearch andcanonicalreadoutfitting,ranbothonGPU,retainedfullfailures,openedallcompletedfilms,andidentifiedexplicitwrong-sign boundarycommands. The broadermotor/utility/multi-flygoalremainsactiveandunfulfilled. Noexternalblockerexists.

### Actual-state wing corrections — 2026-09-13 11:30:20 UTC

This continuation began 11:26:08 UTC. The previous goal turn made progress: two implemented and executed approaches, full evaluations and videos, and explicit startup/wing-limit errors. Retention02 remains selected; no training process was live at the start and there is no external blocker.

The optional correction-capture path relabels an actor's real pre-fall hover states with the existing measured-state reference. It performs offline kinematics only and leaves capture files unchanged. Current source audits retain 116 and 822 frames from the two failed actors, excluding their post-fall histories. Every fifth 20 ms block is held out, leaving actual near-limit examples in training. The original stand/walk/reference histories remain in the fit; startup frames get a declared weight of four. Runtime architecture, body, and graph remain unchanged.

Five focused tests pass in 3.88 s. They include the original and extended full fitting/export path, unchanged upstream/body rows, bounded correction labels with the correct reversing sign at a real canonical wing limit, exclusion after body failure, unchanged source data, and nonoverlapping fitting masks. The full suite is pending at this checkpoint. The new pilot is declared before execution.

The complete motor suite passes 108 tests in 74.52 s, with 23 known dependency warnings. Ruff and diff checks pass. Commit the declared source before executing state_readout02 on the GPU machine.

### Corrected readout outcome and motor-cell feature extension — 2026-09-13 11:45:16 UTC

State_readout02 completes 76.765 s total. Its fit improves, but standing posture still fails and both walking and hover fail the full unassisted test. All outcomes are preserved; retention02 remains selected. The complete film is decoded, inspected and opened. Recorded near-limit fitted outputs still have the wrong sign. This motivates a feature/readout change rather than another unchanged fit.

Fixed repeated NPZ decompression in label extraction. The two real source histories now relabel in 0.104/0.194 s on Mac, with labels bitwise identical to the GPU fit. This is an I/O fix with no physical or target change. Five focused tests pass in 3.94 s. The independent ridge audit initially failed its overly tight cross-host 1e-9 tolerance on a few coefficients; differences are a few billionths. It now records the actual difference with an explicit 1e-8 tolerance and verifies saved coefficient export. No physical gate changed.

Implemented an optional zero-initialized linear wing readout from the same 815 normalized motor-cell activities. It adds 4,896 decoder parameters and no recurrent state. All original actor weights stay frozen. Existing checkpoints keep the original path; the normal loader understands the optional extension. Nine focused tests pass in 5.08 s, including zero-initialization identity, same-history non-wing output preservation, original-weight preservation, learned readout fitting and standard checkpoint reload. Full suite and the declared new pilot are pending at this checkpoint.

The complete suite passes 112 tests in 75.33 s, with 23 known dependency warnings. Ruff passes. The optional motor-cell readout and its declared dataset are committed before remote fitting.

### Motor-cell readout outcome — 2026-09-13 11:58:51 UTC

The optional 815-to-6 linear readout fit completes in 24.764732 s: 5.491297 s setup, 13.887785 s frozen GPU replay, 5.385649 s fitting/packing/save. All original weights remain unchanged; 4,896 new decoder parameters are learned, with no added neural memory, raw-sensor bypass or physical change. Five recorded histories and the same 6,758/1,680 fit/validation frames are reused; zero new integrated experience. Peak allocated CUDA memory is 664,091,648 bytes.

The recorded near-limit fitted command now has the correct positive reversing sign, but its magnitude is still too small. Full unassisted evaluation takes 6.921002 s setup + 31.379153 s capture. Standing remains upright with a failed wing-posture gate; walk and hover fail. First recorded stability-envelope failures are 1.488 s / 0.404 s. All strict task gates fail with zero numerical warnings. Retention02 stays selected. The full15s film is decoded, inspected and opened; render/encode59.486772s.

The source-level decoder tests prove zero-initialization equivalence, same-history preservation of all non-wing actions, unchanged neural memory, bounded wing outputs and standard checkpoint reloading. Independent artifact checks prove every original weight unchanged and exact extension-weight export, matching contracts/hashes and causal actions. Cross-host coefficient recomputation differs by at most3.68e-8 and normal-equation relative residual is1.41e-8, both within float32 source-input precision1.19e-7. An initially tighter helper threshold failed; the physical gates and evaluated checkpoint were unchanged.

Next is a nonlinear decoder readout of the same motor-neuron activities, trained from the existing feature cache and real corrective targets while preserving the parent. No nonlinear extension is implemented yet. This continuation made progress through an executed actual-error data pipeline, a measured I/O fix, an implemented/tested optional decoder, two full physical evaluations and two reviewed films. It did not produce an accepted replacement controller. The full motor/utility/survival goal remains active; no external blocker exists.

### Nonlinear motor-cell readout implementation — 2026-09-13 12:12 UTC

This continuation began 12:07:19 UTC. The preceding goal turn made progress through actual-state correction fitting, two full physical evaluations, and reviewed videos. The intervening sensor-layout clarification changed no implementation. Both hosts are at source63ed79f initially, with no live training job. The broader goal remains active; current priority is stand/walk/hover using the same body and one checkpoint.

Implemented an optional 815→128 tanh→6 wing-logit readout, with training-set feature normalization and zero final-layer initialization. Every original actor weight remains frozen; no new recurrent state or raw-sensor-to-action path. The old linear extension and all unextended checkpoints retain backward-compatible loading. The cached-feature AdamW fit adds no new physics or graph replay. Its declared two-minute limit, weights, split, selection rule and evaluation seed are in runs/nonlinear_motor_readout_01/PLAN.md.

Five focused tests pass in1.39s: zero-output identity, same-history non-wing and neural-state preservation, ordinary checkpoint loading, an actual bounded fitting/export run, exact exported predictions, and tampered-cache rejection. The real five-history cache verifies6758fit/1680validation frames and815features. A one-off Mac audit initially omitted CPU map_location for a CUDA-saved parent; the corrected audit passes. The production fit already explicitly loads to CPU. The complete suite is running; no GPU fit has started yet.

The complete suite passes115tests in75.24s with23known dependency warnings; Ruff and diff checks pass. Source and the declared pilot are committed before remote execution.

### Nonlinear offline outcome and physical-state continuation — 2026-09-13 12:19:37 UTC

The cached-feature fit completes19.509097s total, including1.527266ssetup and17.907381sfitting. It reaches18,000updates, selects10,500, and improves descriptive recorded-history error. The full unassisted test still fails walking/hover and standing wing posture; zero numerical warnings. Every original actor weight, physical contract and causal capture is verified. Exported predictions match acrossGPU/Mac within3.28e-7. The full15s film is decoded, inspected and opened. Retention02remainsselected.

Implemented wing-residual training selection in the existing online motor loop, freezing every original parameter and the feature-normalization buffers. Standard checkpoints now retain optional decoder metadata after online training. Thirteen focused tests pass in21.48s, including actual canonical three-world collection/learning/export with unchanged non-wing outputs on matched histories. The new180s/32world pilot is declared before execution, with purely actor-executed actions and current-state posture/hover labels. Fullsuite is running.

The complete suite passes116tests in77.77s,25dependency warnings. Ruff and diff checks pass. Commit source, offline evidence/video, and the online declaration before remote training.

### Online nonlinear readout outcome — 2026-09-13 12:28 UTC

The real three-minute run completes180.376794s,174,080transitions/348.16aggregate simulated seconds,170updates in32worlds. NativeCPUphysics5kHz,RTX4090brain/learning500Hz. Setup10.172478s;collection174.544957s;optimization5.818723s;peakCUDA896,806,912bytes. The105,222readout parameters change; all original actor state and feature normalization remain bitwise unchanged. Non-wing outputs match the frozen starting actor on174,080identical histories within9.54e-7.

Training retains569completed episodes:58stand(15failures),73walk(40failures),438hover(allfailed). Training timeouts and resets are not physical acceptance. A mistyped graph path caused the first evaluation launch to exit before any rollout; its directory/log are preserved separately. The corrected launch uses the same checkpoint and declared seed72011. Five-second standing/walking stayupright withvalid ground support, butwingposturestillfails;hoverfalls. No numericalwarnings. Setup6.297731s;capture31.530508s. Retention02 remains selected. Fullvideo transfer/render/audit is pending at this checkpoint.

### Archived review — 2026-09-13 12:31:30 UTC

Both completed training films are fully decoded, visually inspected and opened. The online film is15s/750frames/1600×900/50fps/1×;render/encode60.252149s. All493training failure traces and all7500evaluation frames are hash-verified, finite, bounded and causal. Original actor state and feature normalization remain unchanged. The complete116-test suite passes; GitLFSintegrity passes. The motor guide now identifies retention02 and links the current failed diagnostics rather than the older ground-outcome choice.

This continuation made progress: implemented/tested nonlinear readout learning and online decoder selection, executed two measured GPU learning runs, ran two complete physical evaluations, verified and opened both films, and preserved the failures. It did not produce an accepted motor controller. No external blocker exists. The full motor/utility/multi-fly goal remains active. Further work must resolve actual-state wing feedback and flight, rather than treating low recorded fitting error as acceptance.

### Feedback path course correction — 2026-09-13 12:42 UTC

This continuation began12:34:43UTC. Previous goal turn made progress through two implemented/executed decoder experiments, complete physical reviews and verified videos. Mac/GPUstartatcc14700withno active training process. The broad goal remains active; motor learning still has priority.

Frozen nonlinear_online01feedback probe:5.506291s setup +1.875014s diagnostic;75neural sequences,zero physics ortraining updates. Causal replay differs atmost4.33e-7. Next-action angle responses average-0.07152/-0.05204action/rad forstand/walk versus target-0.66667. Speed responses average-0.000717/-0.000405 versus-0.005,with7/8wrong-sign cases outof24each. This identifies weak feedback on these recorded states, not aunique cause.

Implemented an explicit wing-feedback parameter subset:train the existing1792sensor-extension weights and105222wing-readout parameters through the fixed graph. Original sensory encoder, cell dynamics, original motor decoder, allnormalization andphysicalbody remainfixed. Because changed input encoding can change leg commands, a separate ground non-wing distillation loss is available;outputs are never overridden. Two declared optimizer groups allow faster encoder learning. Olddecoder-only contracts remain unchanged.

Eighteen focused tests pass in25.13s. They include actual canonical-world training/export withnonzero sensor-extension gradients/changes, unchanged core parameters andcorrect checkpoint metadata, plus retention-loss gradients confined toground non-wing outputs. Fullsuite isrunning. The180s/32world pilot isdeclared beforeexecution inruns/wing_feedback_01/PLAN.md.

The complete suite passes118tests in79.25s with27known dependency warnings;Ruff/diff checks pass. Source, the frozen feedback diagnostic and the declared learning pilot are committed before GPUexecution.

### Sensor feedback pilot outcome and staged follow-up — 2026-09-13 12:58 UTC

Wing_feedback01 completes180.484524s after10.227206s setup,148,480transitions/296.96aggregate seconds,145updates. Collection146.945133s;backward33.526860s;peakCUDA3,825,026,560bytes. The existing1792sensorweights and105222readout parameters learn; allunselected state andthebody/graph remainfixed. Training has12hover timeouts at2seconds, which are not hover acceptance. All144failure traces areverified.

Full seed72013review:standing/hoverfall,walkingstaysupright butposturefails,allstrictgatesfail,zero numericalwarnings. Setup6.308202s,capture32.135790s. The complete15s film isdecoded,inspectedandopened. A second frozen probe takes4.439183ssetup+1.858781sdiagnostic;replay5.07e-7;immediate ground feedback isstillweak.

The next declared pilot freezes the learned sensory representation andrefinesonlyitsdecoderusingthealreadytestedonlinepath. This testsstagedadaptationwithoutintroducinganotherinput,brain,forcehelperorbodychange. No causal explanation isclaimedfromtheseprobes. Retention02remainsselectedandthefullgoalremainsactive.

### Staged refinement archived — 2026-09-13 13:07:16 UTC

Feedback_readout01 completes180.911450s after10.294863ssetup,178,176transitions/356.352aggregate seconds,174updates. Collection174.575517s;optimization6.324239s;peakCUDA897,061,376bytes. Only105,222readout parameters change. The newly learned sensor encoding, every other parent state entry and the body/graph remainfixed. Non-wing outputs match on178,176identical histories within8.05e-7.

Training has14two-secondhover timeouts among105hover episodes;these areloose-envelope completions,nothoveracceptance. All109failure traces areverified. The unassisted5secondreviewkeepsstanding/walking upright withvalidsupport, butwingpostureandhoverfail;zeronumericalwarnings. Setup5.984868s,capture32.375438s. The complete15s/750frame film isdecoded,inspectedandopened;render59.647187s. Retention02remainsselected.

This continuation made progress through two measured frozen-feedback probes, implemented/tested encoder-and-decoder training with explicit parameter limits and ground distillation, two real GPU pilots, two full physical evaluations and reviewed films. No body, force law, output override or acceptance threshold changed. It did not yield anaccepted motor controller;the broader utility/multi-fly goal staysactive. There is noexternal blocker. Further work must resolve weak immediate motorfeedback andphysicalstability ratherthanrepeatunchangedreadoutfits orclaimsuccessfromtrainingtimeouts.

### Versioned wing position actuation — 2026-09-13

Continuation started at 13:23:09 UTC after a read-only clarification turn. The
old torque-output trials had not produced reliable wing posture or hover.
Implemented `wing_position` alongside preserved `wing_motion`, with explicit
checkpoint migration and a shared physical fingerprint for all three commands.
Only four actuator arrays differ; all other physical arrays and solver options
are exactly equal in the independent saved-model audit. Wings remain massless,
non-colliding and non-aerodynamic, with no inertial reaction on the thorax.

The initial physical reference held the resting wings accurately, but hover
tracking failed at 5.147 mm RMSE. A training-only half-interval velocity lead
compensated the held position target's lag; hover then passed at 1.376 mm over
five seconds. Standing passed. The legacy walking teacher fell in both reference
runs, which are preserved. These demonstrations are not learned-policy results.

Position_feedback01 trained the wing sensory extension and residual decoder for
180.174339 s across 32 worlds, collecting 145,408 transitions. All 193 failure
traces were checked. Its complete autonomous evaluation kept ground wings within
0.353 degrees standing and 0.571 degrees walking, but body posture/tracking and
hover failed. Both ground cases remained upright. The 15 s full review was
decoded, inspected and opened; the preserved checkpoint remains selected.

Inspection exposed a supervision mismatch: retaining the old ground controller
also retained its standing body commands, instead of teaching the full initial
pose requested by the user. Added explicit training-only `--stand-initial-form`
supervision, preserving walking body labels separately. Position_fullbody01
trains the existing motor network and modeled cell dynamics, with immutable
connectome edges and frozen utility/intentions, on the same physical profile.
Its hover collection uses the complete teacher to acquire sustained stroke
histories. Autonomous evaluation is still required; assistance cannot establish
flight success. The arena, utility and multi-agent goal remains active.


### Full standing posture learned — 2026-09-13 13:51:33 UTC

Position_fullbody01 completed 180.165843 s of training across 32 worlds, with
144,384 transitions and 2,336,540 trainable motor/cell-dynamics parameters. The
saved graph identities, utility/intentions, normalization and physical fingerprint
were independently verified; modeled cell gains, leaks and biases received finite
nonzero gradients. The connectome edge multiply has no weight-gradient path.

The complete seed95033 unassisted review passes standing: initial pose held,
0.155 mm root RMSE, no height loss, maximum wing deviation 0.144 degrees. This is
one declared five-second test, not a robustness claim. Walking holds the wings
but produces little forward movement; hover falls. All 27 training failures and
all 7,500 evaluation frames are verified. The 15-second full review is decoded,
visually inspected and opened. Final suite: 124 passed / 85.40 s.

This is progress toward the requested body behavior, with a successful standing
case and unresolved walking/hover. Do not promote a complete motor controller.
Keep the new body fixed while improving actual command tracking and autonomous
stroke feedback; the preserved historical torque-model artifacts remain intact.
The full utility, multi-agent arena and video objective stays active.

### Moving reference and startup correction — 2026-09-13

This continuation begins at 13:55:50 UTC. The prior turn made concrete progress
and left one complete standing pass, with walking and hover unresolved.

A native, no-learning walking probe compared receding and initial-path teacher
previews on the unchanged wing_position body. At the 1 cm/s command, receding
moved -0.453 cm along the initial heading while the anchored path moved 5.001 cm
in five seconds with valid support. At 2 cm/s, the anchored case moved 10.010 cm.
High yaw RMS remains a failed gate. The first probe hit a NumPy-bool JSON error
after saving one complete trajectory; the failed output is retained. Probe02
corrected serialization and saved all four cases.

Added an explicit, training-only anchored preview to the existing batched teacher.
Planar correction is capped at 0.15 cm. Its future path and initial heading are
privileged teacher inputs, absent from the unchanged student schema; therefore
expert tracking is not proof of student tracking. The native/batched equivalence
check passes. Five-second reference review: standing and hover pass; walking has
0.185 mm root RMSE and valid support but yaw RMS 3.091 rad/s fails the old gate.

A replay diagnostic on position_fullbody01 finds a wrong-direction sweep command
at 2 ms: actor -0.356 versus current-state reference +0.217. The first command
at rest is already close, so a correct first output alone does not solve hover.
Added 10x loss weighting for the first 50 ms of hover, without adding an actor
input, runtime controller, oscillator or physical transition.

Position_motion01 uses the same 32-world, three-command body and full motor
network. Walking collection uses the anchored teacher; standing and hover execute
the student. Standing retains the complete initial-pose target. This is a bounded
180 s imitation pilot with fixed connectome edges and utility/intentions, followed
by the original full autonomous gates. Final code suite: 126 passed / 89.01 s,
33 known dependency warnings. The larger survival objective remains active.

### Walking and five-second airborne review — 2026-09-13 14:26:24 UTC

Position_motion01 completes 180.972660 s after 12.127465 s setup, collecting
165,888 physical world/action transitions / 331.776 aggregate simulated seconds,
162 updates. Collection/forward takes 139.260770 s; optimization 41.696225 s;
peak CUDA allocation is 9,420,270,080 bytes. All 152 training failures are hover
cases; all traces are independently verified. There are 30 two-second student
hover timeouts, distinct from the full evaluation.

The complete, unassisted seed96013 review passes standing. Walking advances
4.872 cm along its initial heading over the captured 4.998 s, stays upright with
valid support, and has 1.56 mm root RMSE. Yaw RMS 2.70 rad/s fails the preserved
gate. Flight remains airborne/upright over all five seconds, but overshoots:
initial altitude 1.84 cm, minimum 1.52 cm, maximum 14.74 cm, final 13.16 cm,
backward drift about 3.4 cm. Root RMSE is 89.06 mm. This is sustained flight,
not accurate hovering or a complete motor release.

Evaluation setup is 7.928524 s and capture 31.287544 s, with zero numerical
warnings. All 7,500 evaluation frames pass finite-state, bounded-action and
causal-feedback checks. All graph and physical identities, frozen normalizers,
utility/intentions and declared parameter changes are independently checked.
The 15 s, 750-frame, 1600x900, 50 fps, 1x video is decoded, visually inspected
and opened. Render/encode takes 58.985950 s. Older candidates remain intact.

Next work should check reference recovery from the student's observed high
altitudes before more imitation, then test longer unassisted hover episodes
with normal startup weighting while rehearsing the ground commands. Nominal
reference hover success does not prove recovery over the newly encountered
state distribution. Keep the body and actor interfaces fixed, preserve all
failed gates, and keep the full survival/multi-agent goal active.

### Reference recovery and retained walking — 2026-09-13 14:51:46 UTC

The unchanged measured-state reference recovers eleven of twelve declared
height/vertical-speed starts. The 1 cm / -20 cm/s case touches down and remains
failed. Every world runs five seconds without live resets or numerical warnings.
This supports keeping the physical flight law and reference unchanged.

Position_sustain01 uses five-second episodes and ordinary startup weighting.
After 180.682384 s, hover root RMSE is 20.34 mm but walking becomes stationary.
The failed candidate is preserved. Position_sustain_retention01 returns to the
walking parent and uses a frozen copy as its training-only ground reference.
Every physical action comes from the student; extra weight-4 ground body-command
imitation discourages forgetting. Initial-form standing/resting-wing labels remain.
No physical, sensor, graph or runtime architecture changes are made.

After 181.159775 s, the single frozen actor passes standing, walks 4.84 cm with
valid support and stays airborne for its entire review. Hover root RMSE 6.00 mm
still exceeds the 5 mm gate; walking yaw RMS 2.67 rad/s also fails. Hover ranges
0.90–2.11 cm around a 1.87 cm target. The residual error includes an early dip
and horizontal drift. These two trials share evaluation seed97013 but differ
in supervision and collection; this is not an isolated loss ablation.

All 325 training failure traces, six full command captures, graph/physical
identities and frozen normalization/utility parameters are verified. Both
15-second videos are decoded, visually inspected and opened. The previous
126-test suite covers unchanged training/controller/physics code; this turn
adds a directly executed reference diagnostic. Retention01 is the latest motor
candidate; complete hover, transitions and learned survival utility remain open.

### Bounded PPO diagnosis and observer camera — 2026-09-13 16:28:22 UTC

The user authorized a frozen-actor exploration check and at most two additional
three-minute PPO runs. Diagnostics reveal two limitations: three new hover
starts fail even without noise; on the known successful start .01 noise causes
a fall while .003 noise preserves the five-second rollout. Every parameter
remains frozen in these probes, and paired starts/feedback/hashes are verified.

Outcome03 lowers initial/minimum noise to .003 and still loses hover. Outcome04
adds eight critic-only rollouts before actor updates. It retains airborne
control in the original review and reduces root RMSE from 6.004 to 5.499 mm,
but vertical bobbing is essentially unchanged. It still fails every additional
hover start, including all three deterministic starts; it is not promoted.
Ground stability is retained. No body, graph, reward, actor architecture or
command changes were made. Utility remains disabled. Both checkpoints are
siblings from position_sustain_retention01, which stays the baseline.

The user's camera correction adds a wider view and slower vertical target
following. A new version of the original good video uses identical physical
captures and model hashes. Its existing roughly 5 Hz, 1.204 cm height span is
now visible; the camera had masked that motion. Outcome04 height span is
1.356 cm and vertical-speed RMS 8.20 cm/s versus 8.27 for the parent. Do not
call the lower root RMSE steadier hover or the camera improvement better physics.

The two new pilots take 367.473632 s in total, with 258,048 transitions.
Outcome04 warmup is included (48.194531 s and 64 critic updates). 132 tests and
changed-source lint checks pass. All 386 training failure traces from outcome01
through04 and all 12 evaluation captures are archived and verified, including
the earlier two runs pending archival at the start of this round. Completed
videos are fully decoded, visually inspected and opened. No third PPO pilot
is launched. See PPO_FOLLOWUP.md for evidence and limitations. Further work
must target consistent, quieter hover; practical motor transitions and learned
needs/utility remain unfinished.

### Continuous commands and local hover feedback — 2026-09-13 17:06:35 UTC

Read-only analysis of the existing reference shows a 0.361 mm final-second
height span, compared with 7.282 mm in the preserved learned capture. Its
wing sweep is near 11 Hz versus 10.5 Hz for the learner, whose body oscillates
near 5.25 Hz. Different starts/native hosts prevent a paired performance claim,
but this supports retaining the physical recipe while investigating learning.

The canonical-body continuous test runs stand/walk/stop/resume, two seconds
each, in three worlds without physical or neural resets. All remain upright;
standing/stopping pass, but no world starts walking or resumes. Recorded actor
inputs confirm the correct commands. This exposes the difference between
fresh-start primitives and practical command control. Do not train the transition
worlds solely by copying the frozen parent, which exhibits this failure itself.

Frozen-history hover probes reproduce saved actions within 5.22e-7 and find weak
height sensitivity and some wrong-sign vertical-speed responses. An equivalence
test catches a missing inertial-COM to anatomical-origin velocity correction;
that conversion is fixed before the new teaching loss is used. Probe reports
are preserved with their original source hashes and limitations.

Position_hover_response01 adds local paired-response imitation to the preserved
unassisted recipe, with no body, graph, actor architecture or runtime helper
change. It takes 180.734316 s, 141 updates and 144,384 physical transitions;
2,256 synthetic sensory inputs are counted separately. Standing/walking remain
stable, but the independent hover crosses its height floor at 0.58 s and falls.
All 382 failed training trace windows and all three full captures are verified.
The candidate is not promoted; no extra-start run is needed after this failure.

135 tests pass, with 37 known dependency warnings. Both completed videos are
fully decoded, inspected and opened. The original baseline remains available.
The next practical motor effort should teach command changes inside episodes,
retain fixed-command rehearsal, and preserve the separate hover limitation.
Takeoff/landing, learned utility, and the embodied multi-agent survival task
remain unfinished; this round does not replace those requirements.

### Learning continuous commands — 2026-09-13 17:40:37 UTC

The previous turn produced new evidence and executable diagnostics. This round
implements within-episode command training on the same body and graph actor.
Twelve worlds switch stand/walk each second; five stand, five walk and ten hover
rehearse fixed commands. Switching worlds learn from the current-state inherited
walking reference or initial-form stand targets. Copying the frozen parent on
these worlds would reinforce its failure to start walking. Fixed ground worlds
retain the parent; no new deployed controller or raw-sensor bypass is introduced.

Position_commands01 uses 181.291996 s of unassisted imitation and 143,360 physical
transitions. Its 84 within-episode command changes preserve physical/neural state.
The frozen evaluation now starts walking, stops and resumes in all three worlds,
with approximately 1.9 cm per two-second walking phase. Standing/stopping pass;
moving mean yaw remains above the 0.35 rad/s gate. Fresh-start walking curves and
hover is lost. This is a useful practical gain, not the combined motor solution.

One bounded recovery continuation lowers Adam LR to 1e-5 and blends 80% measured-
state hover reference / 20% student during airborne training. Ground remains
fully student-driven, with the same switching curriculum. Training takes
180.597429 s and 146,432 transitions, with 96 command changes and zero episode
failures. That training success includes assistance: the student still fails
hover when evaluated alone. Ground switching is retained but slightly slower.
Do not select this checkpoint or present assisted survival as learned flight.

Every video uses one checkpoint across all its cases. No successful ground case
is stitched to another actor's hover. Position_commands01 is the useful ground-
transition candidate; position_sustain_retention01 remains the combined reference.
The recovery candidate is preserved as failed. Yaw tracking, robust hover,
takeoff/landing, utility learning and multi-agent survival remain unfinished.

138 tests pass with 43 dependency warnings; source lint passes. Both training
checkpoints, all 366 failed trace windows from the unassisted run, six complete
primitive captures and six complete continuous captures are verified. The guided
run emits no failure windows, so its mixture is evidenced by executed source,
configuration and episode logs; its unassisted evaluation is fully captured.
All four videos are decoded, visually inspected (including fall onsets), and
opened locally. Body/graph identities, frozen normalization/utility parameters,
bounded actions, observed commands and causal prior-action feedback are checked.

### Longer physical-time PPO and sustained hover — 2026-09-13 18:42:30 UTC

After discussion, the user approved renewed PPO work on the existing body and
then a second ten-minute continuation with more resources. We keep native CPU
MuJoCo/mjbatch physics and RTX 4090 neural training, explicitly confirmed during
the run. No Warp port, new body, sensor interface or extra deployed brain occurs.

The first stage replaces short credit settings with 512-action rollouts,
128-action recurrent gradients, gamma .999 and GAE lambda .995 at 500 Hz.
Activation recomputation preserves outputs/gradients while reducing training
memory. Hover learns only from physical outcomes: altitude and velocity costs,
orientation, support and survival. Half the worlds hover and equal quarters
stand/walk; half of hovering worlds gradually receive wider reset perturbations.
Commands remain fixed within episodes in this declared stage. These changes
occur together, so they do not isolate timing as the cause of any improvement.

Timing01 uses 32 worlds and two critic-only warmup rollouts. It takes 606.981945 s
and collects 573,440 transitions, but loses nominal hover. Timing02 resumes all
learning state for 603.603592 s with 64 worlds and no repeated warmup, collecting
819,200 transitions. Combined: 20 min 10.586 s training, 1,392,640 transitions.
No imitation or teacher actions occur in either run.

Timing02 restores the five-second airborne review and sustains one of three
additional ten-second starts; the original parent fails all three matched starts.
The other two candidates delay their envelope violations but still fail. The
user sees the sustained hover as meaningful progress. Bobbing remains large:
14.708 mm full nominal height span, and 9.178 mm final-two-second span in the
sustained additional case. Keep survival separate from accuracy and calmness.
Standing passes; walking yaw remains failed. Preserve this continuation and
its predecessors independently rather than mixing checkpoints across commands.

A frozen neural probe confirms that measured altitude and vertical-speed
perturbations reach wing outputs through the same graph actor. Current/target
height are explicit inputs, not outputs. Response magnitude/direction differs
from the working reference at sampled wingbeat phases; this local probe does
not establish closed-loop stability. The next concrete training limitation is
the shared KL stop: it cuts critic optimization off with actor optimization.
Timing02 gets only 25 updates of each; hover value explained variance remains
negative. Decoupling the critic schedule is proposed, not yet implemented or
claimed to solve oscillation. No third training run occurred in this round.

143 tests and source lint pass. All 584 training failure windows, six nominal
captures, 18 additional task captures and both checkpoint parameter comparisons
are verified. Both full training videos are decoded, inspected and opened.
Measured graph connectivity, physical contracts and frozen utility/normalization
weights remain unchanged. Robust quiet hover, walking yaw, combined command
transitions, takeoff/landing, learned utility and multi-agent survival remain open.

### Independent critic updates — 2026-09-13 19:26:13 UTC

The user asks to continue and fix the critic. A new opt-in path caches its
existing causal input features during physical collection. Policy updates retain
the KL stop; the separate value network completes all two epochs without
replaying the connectome. Saved inputs and GAE targets are detached, and the
deployed actor is unchanged. Tests force immediate policy rejection and verify
continued critic updates, actor isolation, reset-aware feature values and exact
optimizer continuation. Historical commands preserve their original behavior.

Source c455261 is tested and synchronized before the run. Starting from sustained
timing02, critic01 uses the same 64 worlds and all other hyperparameters for
617.449570 s, collecting 851,968 transitions. It performs 26 actor and 208 critic
updates; all 208 value updates occur after the actor's early stop. Independent
value fitting uses only 0.345939 s. The original critic LR3e-4 is deliberately
unchanged to isolate this scheduling correction.

The critic fit improves in 21/26 rollouts but increases error in several late
rollouts. In the last, hover value RMSE rises from 0.3259 to 0.4651 during fitting.
Positive explained variance alone cannot establish accurate values or better
control. A smaller critic step is a plausible next hypothesis, not a proven
explanation for the physical regression.

The frozen nominal review retains standing/walking stability, with walking yaw
still failed. Hover crosses the 0.5 cm floor at1.656 s and falls; root RMSE is
16.324 mm versus7.065 mm for timing02. The candidate is not promoted. Planned
additional-start testing is omitted after this failure, without rerunning the
preserved parent. No second training run or hidden reward/body change occurs.

All 229 failure windows and three complete captures are verified. Parameter
and optimizer comparisons establish unchanged graph/body contracts, 14 fixed
tensors, 18 changed tensors, and exact Adam step increments. 149 tests and
source lint pass. The complete 15-second video is decoded, visually inspected
and opened. Timing02 remains the sustained-hover development checkpoint.


### Hover-first PPO on the accepted PID plant — 2026-09-13

The user accepts the 1 kHz PID reference and authorizes airborne hover-first
learning. The new order is hover, straight/turning flight, stand, landing,
walking, takeoff, continuing one actor. This round trains only hover. Utility
and later motor stages stay deferred. The requested video now directly compares
PID and PPO in separate worlds with the same physical model and nominal start.

We explicitly transfer the old 5 kHz/filtered motor checkpoint to the accepted
1 kHz/instantaneous plant. Two zero-initialized horizontal target-error inputs
extend 397 observations to 399; no encoder bypass or extra brain is added.
The measured graph and routing remain fixed. PPO updates encoder, decoder and
existing neural excitability/leak/bias parameters through physical reward.
The new plant/input contract requires a fresh critic and optimizers initially.
The PID supplies no actions, labels, force targets or wing rhythm to the learner.

Pilot 01 uses 64 hover worlds, 16 CPU physics threads, RTX 4090 neural work,
500 Hz actions and 78 motor outputs. After 607.919 seconds it retains airborne
flight on three ten-second starts, versus two before learning, but drifts almost
6 cm. Its unchanged 602.952-second continuation fails all three. Both are kept.
An audit exposes an incentive flaw: unbounded running costs can make early
failure cheaper than continuing poor flight. We add an opt-in positive bounded
tracking reward, exact -1 failure penalty and explicit critic reset. This is
recorded as a recipe change, not silently applied to historical results.

Pilot 03 rolls back to pilot 01. The bounded reward still fails all three starts
after 306.305 seconds. The KL guard detects large divergence only after an actor
step has already happened; stopping later steps cannot undo the first. Median
observed maximum KL is 1.334 against .03. Pilot 04 repeats the same parent,
seed, fresh critic, reward and five-minute budget with only the actor learning
rate reduced tenfold. Median observed KL becomes .02749; 50 actor updates finish,
versus 18. It stays airborne on all three starts but retains 5.57 mm vertical
bobbing and 59.65 mm nominal peak drift. PID measures .131 mm bobbing and .197 mm
peak error after the first second. That remaining physical gap is not success.

Matching the seed does not make the warmup bitwise identical: tiny initial
reward differences grow, and fourth-rollout episode counts differ by one.
This is one run per setting, not proof that learning rate alone universally
explains the failures. Pilot 04 is a development candidate with less bobbing
than pilot 01 but slightly worse drift. Both remain available; neither replaces
the accepted PID reference or becomes a completed learned-hover release.

All four runs cost 30 min 17.722 s of training and collect 3,866,624 actions,
128.887 aggregate simulated minutes. Setup, evaluation, video rendering and
engineering time are separate. Pilot 04's new-stage ancestry includes 01/04
only; these all inherit an older trained motor actor and are not from scratch.
The full suite passes 174 tests. Four 1x PID/PPO videos preserve startup, drift
and failures, use fixed matched overviews and labeled detail insets, and are
fully decoded, visually inspected and opened. Raw reports, checkpoint hashes,
source revisions, traces and the comparison limitations are archived. No fifth
training run starts in this review round.

## September 13 — smaller exploration and wingbeat diagnosis

Frozen .003/.001 exploration probes and pilot05 are archived with their raw reports. All 192 completed pilot05 episodes survive; mean altitude error improves modestly, height span and nearly 6 cm drift remain. Actor/critic updates are now uninterrupted, but steady accurate hover is not learned. The settled waveform diagnosis distinguishes 500 Hz control from 9.25 Hz learned wing cycles. PID has a programmed 30 Hz rhythm; the actor must generate its own. Low-amplitude disturbances at four wing phases will test response timing before one reward continuation. See hover_only_05 and hover_response_01 run evidence.

## September 13 — phase response and one tighter vertical reward

The user approved a frozen phase-response probe and one PPO continuation after
clarifying that PID and actor both act at 500 Hz. The 30/9.25 Hz values describe
wing motion. Twenty matched branches at four wing phases expose prompt small
command changes (0–4 ms) and lift changes (4–14 ms), but initial correction can
have the wrong sign. All16 disturbances have corrective mean lift by200 ms.
No clock, physics or architecture change was justified by that local probe.

Pilot06 changes only the physical reward's vertical-speed scale50 ->20 mm/s.
The actor/actor Adam/exploration continue; a fresh separate critic receives four
actor-frozen fitting rollouts included in the measured608.933 s. It collects
589,824 transitions on64 hover worlds, with112 actor and144 critic updates.
All192 completed training episodes and three ten-second evaluation starts stay
airborne. Altitude RMS improves2.349 ->2.203 mm, but settled ripple remains2.797 mm,
wingbeat9.25 Hz, vertical-speed RMS about58 mm/s and peak drift about59 mm.
The targeted reward term barely improves. This does not solve accurate hover.

Both new1x videos were fully decoded, sampled frames inspected, and opened on
macOS at23:10:28 UTC. The unchanged settled rhythm is visibly retained. The
original05 checkpoint and all historical results remain. No extra training or
new motor task follows in this review. A better way to learn steadier wing forces,
possibly disclosed exact-plant PID initialization followed by actor-only PPO,
is a discussion option; no runtime helper or imitation was added to this run.
See hover_response_01 and hover_only_06 for raw measurements, timing, hashes,
recipe changes, source code and the critic's remaining weak temporal fit.

## September 13 — exact-plant PID teaching learns a useful wing rhythm

The user approved imitation from the accepted PID followed by PPO. The new
training-only batch teacher reads each current physical state and supplies all
78 joint targets. A 302.066-second, 32-world imitation run uses weight 1 for
wing MSE and .1 for the other posture targets. The PID's explicit phase clock
and integral stay outside the deployed actor. Same graph, body and 500 Hz
actions; all 112 training batches retain target, student and executed commands.

The saved end-of-full-teaching actor independently sustains all three ten-second
starts. It learns approximately 29.75 Hz, 30-degree sweeps instead of 9.25 Hz,
96-degree sweeps. Its settled repeated height ripple is .095 mm, close to the
PID's .101 mm, but mean altitude is about9.86 mm too low and peak position error
is109 mm. The user calls the video promising and likes the smoother motion,
while recognizing the remaining drift and altitude loss. Preserve this exact
checkpoint/video. This is evidence for learning the rhythm, not accurate hover.

The subsequent imitation handoff regresses severely: the final actor falls in
.090/.092/.090 seconds. It is retained with its video and all failed traces.
Low assisted imitation error did not predict independent physical stability.
Under the predeclared selection rule, the surviving earlier actor seeds PPO.
This is development selection, not a held-out comparison. Possible hidden-phase
and state-distribution issues remain hypotheses to test.

One312.290-second PPO continuation uses64 hover worlds,16 CPU MuJoCo/mjbatch
threads and RTX4090 neural work; no runtime PID or imitation labels. Fresh PPO
and critic Adam, four critic-only fitting rollouts within the budget, same
bounded reward, physics, observations and actor. It collects458,752 actions,
46actor/112critic updates and128complete five-second episodes with no falls.
Three frozen ten-second starts survive, but altitude RMS8.724 ->8.695 mm and
peak position error108.999 ->109.575 mm show no meaningful improvement.
The new wing rhythm is retained. No further training is started this round.

The timing question now has stronger evidence: this actor can generate nearly
30 Hz motion at its existing500 Hz action rate. That does not prove adequate
feedback bandwidth or validate the modeled neural time constants. The next
investigation should isolate corrective gain/direction around its own wing
phase before changing clocks or adding more of the same PPO. All179package
tests pass; both imitation videos and both final PPO comparisons were fully
decoded, visually inspected and opened. See pid_imitation_01 and
pid_imitation_ppo_01 for full metrics, hashes, source, clocks and ancestry.

## September14 UTC — longer reward credit does not resolve drift

The user approved the next PPO experiment from the preferred imitation actor.
Rollout length increases1.024 ->4.096seconds, discount decay about2 ->5seconds,
combined GAE trace decay.333 ->2seconds, and episode duration5 ->10seconds.
The399-input/78-output actor, graph, body, reward,64worlds,1,000Hz physics and
500Hz action clock stay fixed. Recurrent gradient chunks remain.256seconds;
this is longer return credit, not longer full-graph backpropagation. The numeric
audit confirms changed delayed-reward weighting and correct episode boundaries.

One620.901816-second run collects1,703,936actions,3,407.872aggregate simulated
seconds,41actor and416critic updates. All320completed ten-second episodes
survive. Four actor-frozen critic-fitting rollouts use135.805563seconds inside
the budget. Eight active rollouts hit the approximate-KL guard; the final round
accepts32updates. The guard limits further updates but retains already-applied
weights and lets the critic continue; it is not a physical fall detector.

Frozen evaluation retains three airborne starts and29.75Hz wing motion, but
nominal altitude RMS8.724 ->8.793mm and peak position error108.999 ->109.484mm
fail the predeclared improvement criteria. The earlier imitation actor remains
preferred. The new actor is preserved as a diagnostic; training time and survival
alone do not establish improvement. Critic explained variance remains near zero
on its own bootstrap targets. This run does not isolate a single cause.

The recommended next discussion is exploration and critic feedback: check
slightly larger action variation on the frozen good actor, verify whether the
critic distinguishes realized better/worse outcomes, then choose a bounded PPO
trial from that evidence. Keep the guard initially; narrow action distributions
can make small mean changes appear large in KL. No new exploration setting or
training run was silently started while reporting this result.

Eleven focused tests pass in7.66seconds; source-identical implementation already
passed179tests. Checkpoint/body/graph/optimizer invariants verified. Both final
videos fully decoded, sampled frames inspected and opened at00:22:29UTC;
20min30s after the observed start. See hover_credit_01 for source, timing,
all measurements, failed progress gates and retained comparison artifacts.

## Exploration and critic scaling — September 14 UTC

A frozen 64-world sweep tests four action-noise levels. All 16 worlds survive
at .003, while ten fail at .006. Critic target scaling alone helps little.
Input standardization improves the mixed held-out set but worsens the safe-noise
subset; both findings are retained. No actor updates occur in these diagnostics.

The subsequent [bounded PPO trial](runs/hover_explore_01/SUMMARY.md) uses .003,
fixed first-rollout critic input statistics and 16 critic epochs. It completes
64 actor updates without a KL stop, compared with heavy early stops previously.
Its 6 min 37 s training still fails both nominal improvement criteria: altitude
RMS 8.724 → 8.652 mm and position RMS 63.708 → 63.776 mm. Three review starts
stay airborne, zero pass accurate hover. Final critic explained variance remains
near zero. More actor updates were achieved; better hover was not demonstrated.
The preferred imitation actor stays unchanged, and value learning remains open.

## Critic sample mixing and six closed flight paths

The matched, fixed-data critic comparison takes 23.105900 seconds without new
physics or actor updates. Shuffling individual time/world samples improves the
safe-cohort median explained variance from 0.0001 to 0.9739 with original inputs.
However, within-time inter-world RMSE worsens and the full predeclared gate fails.
Input standardization performs poorly here. Preserve both results; the critic's
shared time-profile learning does not establish action-outcome discrimination.

The user proposes six paired direction sequences ending where they started.
The [reference check](runs/round_trip_reference_01/SUMMARY.md) uses the same plant
and accepted PID to reach +/-1.5 mm targets, reverse, return and hold. All six
return with under 0.08 mm final-hold error; worst 100ms displacement speed is
0.0307 mm/s. An overly strict 1.5 mm/s raw-speed gate fails even stationary PID
because of small 60Hz wingbeat vibration. Keep that failure; no force or body
filtering was added. This is plant authority, not learned motor behavior.
The same-brain curriculum proposal is in CLOSED_FLIGHT_CURRICULUM.md.

## Closed-flight PPO pilot — September 14 UTC

Implemented the user’s ordered directional trips: origin, one side, opposite
side, origin, hold. Half of 64 worlds remain on stationary hover. Target timing
and distance vary; no future route or clock is passed to the actor. Physics,
graph, action interface and reward weights are unchanged. The critic uses
original inputs and shuffled time/world transitions.

The 400.911823-second run makes 64 actor updates and 1,024 critic updates from
524,288 transitions. Critic in-sample explained variance ends at 0.963, yet the
frozen actor remains unable to follow the paths: 0/7 complete gates, 7/7 airborne,
2.38% worse route RMS than the preferred parent. This isolates a real distinction:
better value fitting does not demonstrate useful target-conditioned correction.
The parent stays preferred; the child and failures are preserved. Full tests:
191 pass. Video shows all seven learned cases plus the separate PID hover.

Next: measure causal target-to-wing response, then consider short PID examples
of correction/braking before more PPO. Do not simply extend the failed trial
or change several reward/physics settings together. Utility remains deferred;
the overall shared-brain survival goal is still active and incomplete.

## Reward objectives before further training — September 14 UTC

User correction: reward the intended behavior, avoid many task-specific fixes.
Implemented one velocity-reference change: requested target velocity plus bounded
position-error correction, tapering to zero near a held target. All old reward
weights and physical/brain settings remain fixed. It is reward computation only,
not an action controller. Same rule for all axes and phases; no new completion
bonus or wing template. Explicit CLI variant and fresh critic on reward transfer.

At equal 8 mm horizontal error, velocity-score ordering changes from stopped >
returning=departing to returning > stopped > departing. This is the specific
incentive being corrected. The old complete-trajectory scores already preferred
the successful PID, so do not overstate the reward issue as proof of all failure.
Nine timing/distance fixture settings across six directions pass all 270 comparisons;
physical PID recordings also score above both learned checkpoints. No new actor
training or new physical capture has run. See flight_reward_01 for all scores,
conservative effort bounds, tests and limitations.

## Goal-directed reward trained and recorded — September 14 UTC

The user requested the missing learned result and video. Repeated the preceding
PPO recipe from the same preferred parent with only the explicit goal-directed
velocity reward flag changed. All 64 worlds, timing, seed, network, critic and
physical settings are matched. Training takes 399.676774 seconds plus 7.961510 s
setup, with 524,288 transitions, 64 actor updates and 1,024 critic updates.
There are five failed training episodes, zero completed tracking sequences,
and no KL guard stops. No teacher or imitation updates occur in this run.

Frozen review remains 7/7 airborne and 0/7 accurate waypoint/return/hold cases.
Six-route position RMS is 75.762 mm: only 0.14% lower than the prior PPO child's
75.870 mm, and 2.24% worse than the preferred parent's 74.103 mm. Stationary
hover RMS is 78.707 mm versus 77.357 mm for the preferred parent. Neither the
progress nor full acceptance gate passes. The new child is diagnostic evidence,
and the preferred parent remains unchanged. This matched single-seed result
does not show that a better local reward preference is sufficient for learning
target corrections. Preserve that distinction instead of claiming the audit
proved a successful controller. See runs/round_trip_reward_02 for measured
resources, physical captures, tensor/config verification and the new video.

## PID movement imitation and rejected target-following review

The user proposes richer expert movements instead of stationary-hover teaching.
Added round-trip PID imitation, fixed teacher assistance and an intermediate
checkpoint to the existing trainer. No new brain, body, actuator interface, PPO
or critic changes. 64 worlds, half hover/half movement, 1.2–1.8 mm excursions and
0.85–1.15 timing scales, full 12-second episodes. Target/input schema documented.

Training takes 601.387001 seconds plus 7.431540 s setup; 133 Adam updates and
544,768 supervised world/action examples. All actions during teaching are PID
actions. All 64 completed teacher trajectories pass their tracking checks.
The five-minute and final actors are evaluated independently without PID or resets.
Route position RMS falls from 74.103 mm to 23.388 mm and then 20.808 mm; hover
RMS falls from 77.357 mm to 24.293 mm and then 21.586 mm. Both stay airborne in
7/7 cases but complete 0/7 precise tracking cases. Both videos are preserved/opened.

The user correctly rejects interpreting this as target-following. Opposite
horizontal requests produce nearly identical paths; vertical response is reversed.
This exposes a limitation of aggregate error as a progress signal: a less-drifting
oscillator can score better without correctly obeying commands. Preserve the
numeric gates and their outcomes, but do not accept the policy as a controller.

A zero-step read-only audit verifies tested target encodings, unit scales,
world/body rotation and PID correction signs. No corresponding wiring error is
found in those fixtures. Saved teaching data has 95th-percentile altitude error
0.1508 mm and horizontal error 0.9464 mm, versus student drift of tens of mm.
Actual/requested teaching altitudes correlate 0.9841. Richer target paths still
leave most expert states close to target and do not guarantee learnable recovery.
The exact cause of reversed learned response is not fully established. Next work
must validate corrective response on matched error/recovery states before more
training. See pid_movement_imitation_01/SUMMARY.md and REVIEW_DECISION.json.


### PID imitation: implementation audit after rejected movement review

The user requests a deeper bug investigation: walking imitation worked and the
PID controls the same plant, so more training/data should not be assumed to be
the answer. Freeze physics and actor weights. The real-graph GPU audit verifies
wing actuator channels/normalization, reproduces the parent's original recorded
training predictions, and compares deployment and training forward/backward
paths. All tested mappings and gradients pass; current/target position inputs
receive gradients. Twenty relevant existing physical/control-path tests pass.
No new learning updates. Root cause remains unresolved; target following remains
unaccepted. Preserve this distinction in demos and avoid presenting drift
reduction as flight-command learning.

The user reaffirms that the aim is consistent, learnable dynamics, not realistic
aerodynamics. Do not introduce realism qualifications as an issue with the chosen
plant. Half of the 64 training worlds hover; the others execute separate 12-second
out-and-back directional exercises in the same run, updating the same actor.
Evidence: runs/pid_movement_imitation_01/learning_path_audit.json and SUMMARY.md.

## Fresh velocity interface and continuous PID exercise — September 14 UTC

The user requests a fresh start without inherited policy, optimizer or fitted
normalization. Keep the measured MaleCNS graph and use velocity/turn-rate inputs
instead of position targets. Every episode must contain all skills together;
do not partition hover and directional tasks across separate worlds. First
demonstrate the teacher and physics before any new student training. The earlier
imitation failure remains unresolved and archived; this change is not proof
that data coverage alone caused it.

Independent heading exposed a concrete force-law limitation: sweep imbalance
coupled roll and yaw. The explicit wing_motion_heading_v3 extension adds yaw
authority from measured wing pitch difference, keeping the same physical body,
limits and 1,000 Hz physics /500 Hz controls. It reads no commands or targets.
All forces still come from measured wing/body state at every physical tick.
Previous physical contracts and checkpoints remain preserved.

Three PID captures share the exact compiled model. Feedback tuning improves
declared stage checks from 29/50 to 49/50 to 50/50. The final 71.2-second episode
has no resets; isolated strafing changes heading by at most 0.123 degrees.
Translation, turns in place, mixed commands, brakes and hover all appear in
the full video, decoded, sampled and opened for review. Final settled errors
are below 0.5 mm/s translation and 0.12 rad/s yaw gates using a documented
100 ms mean; raw wingbeat velocity is retained and shown separately.

The fresh encoder/decoder initialization path is implemented and tested on a
small graph fixture, but a full fresh checkpoint has not yet been instantiated
and no student training occurred. The user questions adding external utility
and navigation networks: that would outsource decisions they want learned
through MaleCNS. Keep one shared core as the intended path. A future utility
readout may expose its activity, but separate decision networks are not the
default plan. Frozen wiring alone does not establish that the core is necessary;
future ablations should test its contribution to learned behavior.

Evidence: runs/pid_velocity_teacher_01 through _03 and VELOCITY_TEACHER.md.

The user reviewed the completed video: "the video looks great," then requested
approximately ten times faster physical flight, explicitly keeping real-time
playback. Preserve this slow reference and develop a separate faster capture;
the next work is still the PID/plant, not student training.

## Ten-times-faster physical flight — September 14 UTC

The user explicitly wants faster dynamics, not accelerated footage. Command
translation increases from 1.5 to 15 mm/s per axis and yaw from 0.45 to 4.5 rad/s.
The same 71.2-second continuous exercise, 1,000 Hz physics, 500 Hz controls and
1x video clock remain. Every physical command is sent through bounded wing joints.

The original heading model's angular resistance prevents the requested fast yaw.
Explicit v4 keeps roll/pitch damping and changes only body-axis yaw resistance
from a 0.025-second to a 0.25-second time constant. Body mechanics and force gains
are unchanged. The first fast capture remains stable but passes 14/50 stages.
Causal command-acceleration/drag feedforward and stronger PI feedback increase
this to 25/50; tight sideways turns still fail. More feedback alone is insufficient.

Explicit v5 adds lateral thrust from measured wing-stroke angle difference,
bounded at 25% of instantaneous lift. This gives independent side thrust while
turning instead of relying entirely on banked lift. No commands, goals or PID
state enter the force model. No wing activity means zero flight wrench. The
v5 capture passes every translation check, with 34/50 combined stage checks;
remaining failures are small yaw overshoots after stopping. Increasing yaw
proportional gain to 60 and reducing integral gain to 30 yields 50/50 on the
identical v5 compiled model. All unsuccessful development captures are retained.

The final reference measures 15.000 mm/s in isolated translations and about
259.5 degrees/s in isolated turns. Worst settled vector-speed error is 0.5643 mm/s
and yaw-rate error 0.0591 rad/s. Faster-motion limits were declared before these
captures; stop/hover limits remain the original 0.5 mm/s and 0.12 rad/s. The
same upright, altitude and contact gates pass. These are development checks on
one prescribed episode, not a claim of broad robustness.

Four complete captures collect 142,400 actions and 284.8 simulated seconds,
with zero neural training. The first report writer failed after saving its
trajectory due to a NumPy boolean serialization error. The bug was fixed and
the report recovered from saved states; its lost compute timer is marked unknown.
The other three captures total 195.151927 seconds; final capture 65.779003 seconds.
See runs/pid_velocity_fast_01 through _04 and VELOCITY_TEACHER.md.

The first faster render was decoded, sampled and opened, then its distant
overview was replaced by a fixed-scale XY map using the same recorded frames.
The map shows actual past motion and current heading, with a 5 mm grid and a
clearly labeled position marker. Final v3 remains 71.2 seconds at 1x, fully
decoded and visually reviewed, opened 2026-09-14 05:03:41 UTC. No new physics
or learning was run for this presentation fix. The previous videos are preserved.

## Fresh velocity imitation, first one-minute burst

The user requested 60-second learning bursts with continuation state preserved.
The new replay trainer uses the fresh 391-input velocity actor and exact accepted
v5 flight model. Ten complete 71.2-second PID demonstrations pass all 500 stage
checks; eight train and two validate. All movements remain in every episode.
Sampling recurrent windows from these recordings prevents a short wall budget
from seeing only the early stages. Every update covers all 50 stage IDs.

The first 62.554799-second burst completes 14 Adam updates and 114,688 supervised
targets on the RTX 4090, with 64 sequences, 128 supervised steps and 64 context
steps. Checkpoint, optimizer and sampler state are retained. All encoder/readout
and cell-dynamics paths receive gradients; graph weights stay unchanged.

Held-out wing MSE falls from 0.396961 to 0.008018, but both student-only physical
episodes fail at 0.070 seconds, before any movement command. The read-only audit
finds that constant mean wing positions achieve lower MSE (0.004135), and the
student's predicted sweep variation is much smaller than the teacher's. The
large loss reduction is therefore not evidence of learning the wingbeat. Fourteen
updates are too few to settle whether sustained imitation will learn it.

The failure is visible in the synchronized 1x comparison, opened after QA. No
extra burst was run. The user subsequently requested a proposed configuration
for a 30-minute training allowance; this discussion does not itself constitute
a completed longer run. See runs/velocity_imitation_01/SUMMARY.md for measured
times, preserved checkpoints and exact continuation commands.

## Continuous thirty-minute velocity imitation

The approved 1,800-second continuation finished 401 additional updates in
1,803.594666 seconds. Same checkpoint, Adam state, graph, physical model and
loss; the explicit sampler change uses eight cold starts plus 56 all-stage
windows. Held-out wing MSE drops 94% to 0.000483. Sweep correlations reach
0.989/0.990, beating the constant-pose baseline, and measured startup wing
speeds approach the PID. This supports learning wing motion, not successful
flight: both fixed physical evaluations climb toward 99 mm and fall after
5.624/5.626 seconds. No completed tracking stage passes.

The user-requested midpoint CPU peek found upright but climbing behavior over
one second; it did not interrupt GPU learning or determine checkpoint choice.
The final result uses the last checkpoint. A NumPy report serialization bug
required repeating evaluation, without more training. All 214 tests pass.
The 1x video was decoded, visually inspected and opened at 07:02:45 UTC.

All ten teacher episodes used the same sequence; headings/wing phases varied.
The user approved varying order, retaining all skills in each complete episode,
and collecting PID corrections from student-visited drift/climb states before
continuing the same policy. See runs/velocity_imitation_30m_01/SUMMARY.md.

## Reordered PID recoveries and second full continuation

The user approved varied ordering, correction from student-visited states and
continuation of the same checkpoint. Ten PID starts recover from eight actual
student snapshots plus two cold starts. Every two-second recovery gate passes.
Ten complete reordered episodes then pass 500/500 stage checks, with no live
resets. Combined with original data, 16 training/four validation episodes pass
1,000 checks. Per-episode semantic stage sampling and explicit dataset-transfer
permission prevent accidentally sampling the old timing on reordered recordings.

The second continuous block performs 401 updates in 1,804.362681 seconds, taking
the actor's ancestry to 816 updates and 61 min 10.5 s. Same network, fixed graph,
physical plant, optimizer state and loss. No PPO, critic or old reward functions.
Held-out wing MSE improves 13.1% on the mixed data, and the original-window audit
also improves. Physical flight regresses: all four cases fail opening hover at
0.202/0.202/0.202/0.244 seconds. Initial observations match recorded starts exactly.
Wing sweep speeds fall from about 31 rad/s at startup to 17–18 rad/s during
0.15–0.20 seconds; the teacher and parent remain near 31 rad/s in that window.

Do not treat better offline command matching as a successful controller. These
are complete teacher recoveries branched from saved student errors, not online
DAgger throughout student rollouts. Retain the parent (5.6-second airborne
flight) as the better development reference; it also remains unsuccessful.
No further training was launched. The 27-second, four-case 1x comparison was
decoded, inspected and opened at 07:54:34 UTC. Final and all periodic checkpoints
are preserved. Full suite 218 passed; two new random-command tests passed.

The user's smooth random command idea now has a separate seeded generator with
quintic transitions, bounded commands, isolated/combined requests and zero holds.
It is not included in this block. A future physical collector must bound altitude
and validate its teacher trajectories before using that additional data source.
See runs/velocity_recovery_imitation_01/SUMMARY.md and RANDOM_COMMANDS.md.

## Velocity hover PPO pilot and update overshoot

The user approved a ten-minute PPO pilot from the first 30-minute imitation
parent. Keep the same 391-input actor and agile flight plant, use 32 hover worlds,
new velocity/rotation/support rewards, a fresh critic and light PID hover replay.
The actual GPU recurrent audit passes after accounting for float32 error summed
across 78 low-variance action channels; no weights updated during the initial
overly tight audit stop. PID and parent were evaluated once on uninterrupted
ten-second hover. Parent survival is 10/6.772/10/4.190 s, distinct from the prior
mixed-exercise result. PID holds all four, with settled error below 0.063 mm/s.

Run 01 completes 604.367622 s, 1,081,344 transitions, 65 actor updates and 2,112
critic updates. The midpoint falls at 2.434 s; final falls at 0.470 s in all
four cases. Temporarily longer noisy training episodes and accurate fitted
critic targets do not establish improved deterministic control. The KL monitor
stops after one actor minibatch per rollout but retains the oversized step
already taken. Do not call this a successful policy or conclude PPO cannot work.
Final/midpoint weights and the 1x three-way video are preserved. Video opened at
08:43:53 UTC after full decode and visual inspection.

User steering authorizes continued improvement if this pilot fails. Run 02
returns to the original parent and uses smaller initial LR plus analytic
post-step KL acceptance, exact parameter/Adam rollback and backtracking. Rewards,
body, observations, commands, worlds, teacher anchor and critic are unchanged.
See runs/velocity_hover_ppo_01/SUMMARY.md and runs/velocity_hover_ppo_02/PLAN.md.


### September 14 — bounded PPO, reward scale, focused readout

Run 02 fixes optimizer acceptance: all 146 accepted steps stay below analytic
minibatch KL .02. It slightly prolongs two flights but increases drift/climb.
Run 03 changes only horizontal reward width .5 -> 2 cm/s. Final survival is
10/10/7.232/4.732 seconds; early speed RMS barely changes (25.96 -> 25.88 mm/s).
Neither passes the declared hover milestone. Keep both checkpoints and videos.

The user explicitly authorized continuing improvement rather than stopping at
another failed pilot. Run 04 holds encoder/core/base decoder fixed and trains
the existing 105,222-parameter wing readout. All actor inputs/outputs and physics
stay the same. Cached recorded motor-neuron features give exact conditional PPO
replay while upstream weights are frozen. Tests verify equality after updates
and resets; the full-GPU smoke verifies action/log-probability reproduction and
zero upstream parameter changes. Exploration is held at .003 during this stage.
This is a temporary trainable scope in the same actor, not a new controller.

Full fly suite after this implementation: 229 passed, 45 known upstream warnings,
165.88 seconds. Run 04 remains subject to the same physical evaluation criterion.


### September 14 — four completed flights, with a climb tradeoff

Run 04's final actor completes all four ten-second captures. Early velocity RMS
improves 25.96 -> 20.28 mm/s (22%). In paired complete case 0, horizontal RMS
improves 30.68 -> 18.83 mm/s and full RMS 31.77 -> 24.40 mm/s. However vertical
RMS grows 8.23 -> 15.51 mm/s, net climb 44.68 -> 152.00 mm and peak displacement
148.85 -> 217.18 mm. Report both outcomes; no stable-hover claim.

The initial survival/early-error milestone is reached. Because upward drift
still needs regulation, continue this same final checkpoint in run 05. Restore
actor Adam, trained critic/Adam, fixed exploration and actor sampling RNG.
Physical episodes restart at the same declared starts; critic shuffle alone
restarts from the recorded seed. A discarded 17.336441-second GPU smoke verifies
restoration, no critic recalibration and no upstream parameter changes. Preserve
the improved run 04 independently in case continuation regresses.

The readout cache is training-only: full graph processing occurs in every live
physical actor step. The speed gain is from reducing trainable scope and avoiding
repeated full-graph backprop, not from moving physics to GPU or bypassing the
connectome. Same deployed architecture throughout.


### September 14 — continued PPO improves full flight; altitude remains open

Run 05 preserves all four ten-second completions and reduces velocity RMS from
24.4 to 14.92 mm/s, horizontal RMS 18.83 -> 9.30 and vertical RMS 15.51 -> 11.67.
Net climb decreases 152 -> 106 mm and peak displacement 217 -> 117 mm. Against
the original parent on the two matched complete flights, total velocity RMS is
53% lower, horizontal RMS 70% lower and peak displacement 21% lower. Vertical
regulation remains worse than that original parent's approximately 45 mm climb.
The initial sustained-flight/reduced-error pilot milestone is met; do not label
this stationary hover or a broad robustness result. RMS windows are specified
in HOVER_PPO_PROGRESS.md.

The final state is selected by the planned end-of-budget evaluation, not by
searching the video for the best moment. Both midpoint and final weights and
trajectories are preserved. No reward, physical model, policy architecture or
trainable scope changed during this continuation; actor Adam and critic/Adam
were restored. All upstream actor weights remain unchanged in the two productive
PPO stages, while the full graph remains in the live control path.

Both the continuation and the direct original-to-final comparison were decoded
fully, inspected and opened. The final checkpoint is a useful starting point
for vertical regulation, not a completed release of all fly motor skills or
utility learning. Total trial cost and selected ancestry are kept separate.


### September 14 — unchanged continuation plateaus; explicit vertical adjustment

Run 06 completes four ten-second flights, but horizontal RMS worsens from
9.30 to 15.43 mm/s and net climb only decreases about 106 to 101 mm. Its midpoint
also trades sideways control for vertical improvement. Preserve both snapshots,
but retain run 05 as the better parent. Exact recipe equality and unchanged
upstream actor parameters are verified. The comparison was fully decoded,
inspected and opened at 15:15:04 UTC.

The user-approved conditional fallback is run 07: resume run 05 and increase
only the vertical tracking reward rate from 2 to 3. Keep the same velocity width,
all other rewards, physics, commands, network, exploration and PPO settings.
One initial rollout adapts only the restored critic to the new reward; do not
recalibrate its inputs or discard its optimizer. A short discarded GPU smoke
verifies that transition, later actor updates, cached/full recurrent replay and
unchanged upstream parameters. Full fly suite: 231 passed in 165.98 seconds.
A launch-path typo aborted before graph loading completed or any training;
the aborted output is preserved separately and the path corrected.


### September 14 — the vertical reward works, but trades away sideways control

Run 07 final completes all four flights and lowers net climb 105.99 -> 67.41 mm
and vertical RMS 11.67 -> 7.43 mm/s (both approximately 36%). Horizontal RMS
worsens 9.30 -> 14.07 mm/s (51%); total RMS worsens 14.92 -> 15.91 mm/s and peak
displacement 117.10 -> 126.85 mm. The midpoint trades the other way: horizontal
RMS 7.21 mm/s but climb 119.30 mm. Neither snapshot passes the joint criterion.
Do not promote an axis-specific gain as better overall hover. Keep run 05 as the
preferred overall checkpoint, and preserve the new vertical-control candidate.

The recorded recipe differs only in the vertical rate, derived maximum/version,
and explicitly declared critic adaptation. First-rollout actor updates are zero;
critic calibration is retained; all 413 accepted actor updates meet KL .02.
Full and cached replay pass, and all upstream parameters remain identical.
The 43-second comparison was fully decoded, inspected and opened at 15:32:55 UTC.
The next issue is simultaneous vertical/horizontal regulation; stationary hover,
commanded flight and later utility behavior remain unfinished.

Public architecture clarification: sensory input and motor readout use annotated
MaleCNS cell classes, without a raw observation-to-action bypass. The learned
motor decoder is not constrained to a verified per-muscle innervation map.
Fixed graph routing alone does not establish that specific biological circuits
are necessary; that would require targeted ablations, not just a performance video.
