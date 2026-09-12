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
It is a hypothesis, not an established cause of every failure.

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
