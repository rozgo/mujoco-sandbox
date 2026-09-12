# Course correction: train the embodied fly controller

Research checked September 12, 2026. This is a proposal, not an implemented
replacement. The utility-only training effort is paused; its source, generation
checkpoints, physical captures and inspector work are preserved.

The user's revised objective is one trained neural system that learns locomotion
and activity selection, potentially through a curriculum. Training a small
activity selector above an existing gait controller does not meet that objective.

## Recommendation

Use the **FlyBody anatomical model and physical flight machinery**, a **trainable
sparse recurrent controller constrained by MaleCNS v1.0 connectivity**, and
**imitation followed by reinforcement learning**. Preserve the same actor across
walking, flight, transitions and survival stages. Start by proving learned
six-leg locomotion with the external gait controller completely disconnected.

FlyGym itself does not prohibit learned control. Our decision to use its supplied
hybrid walking controller was the limiting architectural choice. FlyGym 2.1 also
contains experimental FlyBody integration. The body and controller must be
assessed separately. Its FlyMimic implementation currently actuates only the
left-front leg with muscles; it is not a complete muscle-driven fly ready for
free locomotion. [Official FlyGym documentation](https://neuromechfly.org/tutorials/6_muscle_imitation/)

## What the research establishes

| Source | Useful evidence | Remaining gap for us |
|---|---|---|
| [FlyGM, June 2026 revision](https://arxiv.org/html/2602.17997v3) | A connectome-structured controller learns body actuation from expert demonstrations and PPO | Uses FlyWire; walking and flight have different interfaces; flight retains a wingbeat generator |
| [FlyGM project page](https://lnsgroup.cc/research/FlyGM/) | Demonstrations and architecture description | Code still labeled “Coming soon” when checked; no verified downloadable reproduction |
| [DeepMind/Janelia FlyBody](https://github.com/TuragaLab/flybody) | Anatomical body, walking/flight tasks, published data and trained reference policies | Its supplied learning dependencies are old; a unified walking–flight controller is not supplied |
| [Pugliese and colleagues](https://pmc.ncbi.nlm.nih.gov/articles/PMC13142387/) | A preprint identifies candidate walking rhythm circuits in VNC connectome simulations, with experimental evidence for a descending pathway | Motor rhythms are not demonstrated free whole-body control |
| [Flyhard pilot](https://github.com/MarkUnthank/flyhard/blob/main/docs/pilot-2026-09-09.md) | An actual MaleCNS-derived core learns one foreleg steering skill; repository reports 186 seconds and about 3 GB on an A6000 | Supported thorax, artificial grip, stationary target interpolation; not walking or flight |
| [2026 full-body tracking dataset](https://www.janelia.org/publication/whole-body-3d-kinematics-of-freely-behaving-drosophila) | Natural terrestrial behavior measured at 800 fps with 50 keypoints | Retargeting and suitability of individual behaviors still need checking |

FlyGM is the closest algorithmic precedent, not an off-the-shelf MaleCNS solution.
Its paper describes fixed connectivity with learned internal cell descriptors and
updates; training the internal computation is different from training only an
output selector. It reports A100 80 GB hardware. We have not reproduced its
results or established its runtime on our 4090.

The new MaleCNS dataset includes the ventral nerve cord. Our downloaded annotation
table contains **708 `vnc_motor` and 6,370 `vnc_sensory` entries**, along with
13,161 `vnc_intrinsic` entries. Motor annotations include limb and wing groups,
nerve labels and some muscle-related names. These are useful routing clues,
not a complete calibrated motor-neuron-to-MuJoCo-actuator mapping.
[Official data](https://male-cns.janelia.org/download/)

## One integrated actor

Proposed deployed loop:

```
Eye images / odor / touch / joint feedback / internal needs
                         ↓
                  learned sensory encoding
                         ↓
           MaleCNS-constrained recurrent neural state
           sensory + central + descending + VNC pathways
                         ↓
             learned motor-neuron-to-actuator decoding
                         ↓
          joint targets or torques + adhesion + wing torques
                         ↓
                  physical MuJoCo body
                         ↓
                     new sensations
```

All flies share one set of learned parameters and one deployed checkpoint. Each
fly keeps separate recurrent state and needs. Internal modules may update at
different rates; that does not require separate policies selected by Python.

Keep graph topology and neuron identities auditable. Begin with low-dimensional
neural states and train parameters **inside** the graph: bounded connection gains,
cell gains/leaks and sensory/motor interfaces. Maintain temporal state instead of
resetting the network on each decision. Preserve transmitter sign assumptions
explicitly; neither transmitter prediction nor synapse count uniquely determines
physiological dynamics. Prevent a dense sensory-to-action skip path from bypassing
the graph.

A practical first configuration to benchmark is 1–4 state channels and 4–16
parallel worlds with short recurrent training sequences. These are proposed
measurement points, not measured capacity. Avoid constructing per-edge feature
messages or dense N×N matrices. Flyhard's source demonstrates a chunked sparse
edge-gradient implementation after ordinary sparse backward tried to allocate a
large dense intermediate; reuse requires gradient checks and a license audit.
[Implementation](https://github.com/MarkUnthank/flyhard/blob/main/src/flyhard/connectome.py)

For locomotion, bounded position actuators are a reasonable initial muscle
abstraction: the network selects individual joint targets; MuJoCo produces
limited forces and contact. Wings need physically modeled aerodynamic forces.
Detailed biological muscles can be added later, but must not be claimed from a
joint-servo implementation.

Replace the external six-way utility winner with learned internal motivation or
skill state that influences the same motor-generating network. Needs enter as
observations and determine outcome rewards. If explicit utility scores are kept
for interpretability, their selection mechanism must also be trained within the
actor. The UI must distinguish an actual decision head from a post-hoc behavior
classifier. A training-only critic estimates returns; it does not supply rewards
or control the fly at deployment.

## Curriculum and acceptance

These stages extend the same actor. All actuator slots and observation fields
should exist from the start; masks, initial states and reward emphasis change
between stages. Old motor skills remain in the training mix as later skills are
introduced. That rehearsal is part of training time and will be reported.

1. **Graph-to-body proof.** Learn a bounded joint-reaching/contact task from
   demonstrations. Verify finite, nonzero core gradients and changed core weights.
   Compare frozen-core and trained-core models with identical interfaces. This
   is a diagnostic prerequisite, not the final locomotion demo.
2. **Free six-leg walking.** Use recorded fly movement and/or a reference learned
   controller to initialize the same graph actor. Learn standing, starting,
   stopping, speed changes and left/right turns. During evaluation, remove all
   teacher calls, reference trajectories, supplied gait phases and hybrid-controller
   outputs. Require free-body motion, actual foot contacts and bounded actuation.
3. **Feedback and recovery.** Add pushes, changed friction, modest terrain and
   perturbed starts. Use closed-loop imitation/data aggregation if the student
   visits states missing from the demonstrations, then PPO for physical outcomes.
   Preserve healthy stepping while learning corrections.
4. **Wing control and transitions.** First learn flight-compatible wing/body
   control; then takeoff, landing, folding and unfolding. Use the same physical
   model with legs and wings present, not separate bodies that delete unused
   degrees of freedom. Reference wingbeats may teach initialization, but the final
   test must remove the external wingbeat generator if we claim learned wingbeats.
   The recurring neural state must then generate the motor rhythm. If this fails,
   report flight as assisted or incomplete rather than relabeling it as learned.
5. **Needs and useful activity.** Add hunger, hydration, fatigue and threat to the
   same observation schema. Learn resource seeking, a physical intake posture,
   shelter, escape and recovery while retaining locomotion training. Use longer
   evaluation horizons than the depletion times to expose camping shortcuts.
   Grooming, proboscis control and courtship require appropriate demonstrations,
   body interfaces and measurable objectives; they are not automatic consequences
   of a locomotion policy.
6. **Social environment.** Put independent copies of that one actor in the shared
   habitat. Begin with modest interactions, then finite resources, crowding and
   hazards. Retain all predetermined trial outcomes. Show recurrent activity,
   motor outputs, needs and the actual selected fly on a common clock.

A learned teacher can help initialize the student without becoming a deployed
body controller. Teacher-specific future reference trajectories are privileged
training information, not final sensory input. If the teacher is reused from an
upstream project, report its provenance and distinguish inherited motor knowledge
from our training time.

## Compute and software decision

Use the RTX 4090 for neural forward/backward passes. Compare current MuJoCo Warp
with native CPU MuJoCo for the **same complete graph–body learning loop**, not a
physics-only hold. Keep observations, rewards and neural state on GPU on the Warp
path. Measure eye rendering separately and add it after the motor prototype.

Current MuJoCo Warp supports fluid-related machinery and batch rendering; old
claims that its ellipsoid fluid model is universally unsupported are stale.
Actual wing-force, adhesion, collision, timestep and native/Warp agreement tests
remain required for the selected versions and model.
[Current Warp documentation](https://mujoco.readthedocs.io/en/stable/mjwarp/index.html)

FlyBody's present optional learning stack pins TensorFlow 2.8, Acme and related
older packages. Use the body assets/data as references and a new isolated,
uv-locked current PyTorch training stack. If necessary, use an isolated legacy
**teacher export** environment only to reproduce and convert supplied policies;
it should not become the runtime or new training dependency.
[Verified source](https://github.com/TuragaLab/flybody/blob/d015e9bfe441bd90ae431bac24c55cb74bdbce26/pyproject.toml)

The upstream defaults are 500 Hz walking control with 5 kHz physics, and 5 kHz
flight control with 20 kHz physics. These rates make a full-brain-per-tick flight
controller considerably more expensive than our former 50 Hz utility selector.
A fast VNC/motor part and slower sensory/central updates within the same network
are a proposal to benchmark, not validated biological timing.
[Constants](https://github.com/TuragaLab/flybody/blob/d015e9bfe441bd90ae431bac24c55cb74bdbce26/flybody/tasks/constants.py)

A full fly in five minutes is not a supportable promise. The original FlyBody
work reports roughly 10^9 simulation steps for walking and 10^8 for flight,
with training measured in hours or days on its distributed setup. Those figures
are context, not a forecast for a new implementation.
[Published training](https://www.nature.com/articles/s41586-025-09029-4)

Keep individual pilot training runs at five measured minutes, extending to ten
only when learning curves and held-out motor behavior justify it. Report setup,
data collection, supervised updates, RL, evaluation and rendering separately.
Choose world count after measuring throughput and memory; do not import the dog's
4,096-world count into a 166,700-neuron recurrent model without evidence.

## Evidence that would justify the story

- The actor controls individual body actuators with the external gait/WPG paths
  disconnected at evaluation; no base pose assignment or hidden support.
- Learning changes internal graph parameters; freezing them while allowing the
  same interfaces to train provides a meaningful control.
- Graph removal, selected pathway interruption, delayed sensory feedback and
  degree-preserving rewiring have explicitly measured effects. A rewired model
  should be trained under matched budgets for a topology-benefit claim.
- One checkpoint retains walking, stopping, turning and later flight/survival;
  transitions do not switch to separate policy files.
- Unseen starts, targets, hazards and physical perturbations retain full failures.
- Anatomy colors show model state, labeled as rate/latent activity or spikes
  according to the actual neuron model. Need charts show actual internal state.

The defensible target is **a trainable, connectome-constrained model that learns
fly-like embodied behavior**. Restoring the original animal's brain would require
unmeasured physiology, sensory calibration, neuromuscular mapping and validation
against biological recordings. Functional motor learning is attainable as an
engineering research target; a recovered real animal is not established by it.

## First deliverable after approval of this direction

A single free fly learns to stand, walk, stop and turn with the trainable MaleCNS
controller driving limb actuators. Deliver matched before/after videos,
core-gradient/parameter evidence, held-out physical metrics and measured 4090
runtime. Preserve the habitat and inspector for reuse, but stop polishing or
training the former utility-over-scripted-gait architecture as the main result.
