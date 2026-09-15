# Evidence to retain before proposing more fly diagnostics

Reviewed September 15, 2026, 01:54:56 UTC, following the user's reminder to consult
the journals. This is a documentation review: no new simulation, optimization,
checkpoint selection or video. The earlier recommendation to first test whether
velocity feedback reaches motor neurons repeated a completed study. Treating a
missing sensory representation as the leading explanation overlooked evidence.

## Questions already investigated

| Question | Existing evidence | Scope and implication |
| --- | --- | --- |
| Do independent velocity signals reach the motor cells? | [Velocity PPO 08](runs/velocity_hover_ppo_08/SUMMARY.md): forward/left/up inputs perturbed by ±0.1 cm/s over 2–100 ms at three recorded ages. Independent motor-feature responses; derivative singular values about 0.26–0.39 at 100 ms; action replay within 3.58e-7. | Current agile-v5 family, run-05 ancestor. Rules against disconnected or collapsed local velocity inputs on those histories. Does not prove global representation sufficiency. |
| Does the actor respond to disturbances at different wingbeat phases? | [Hover response 01](runs/hover_response_01/SUMMARY.md): 16 physical disturbance branches, commands change within 0–4 ms; correctly signed sustained lift response in 4–60 ms. All 16 average lift differences have the correct sign over 200 ms, with initially wrong directions in some cases. | Earlier hover plant/checkpoint, before agile-v5. Completed phase-response study, not a present-checkpoint bandwidth guarantee. Do not propose it as an entirely untried investigation. |
| Have encoder and decoder learned together? | [Velocity imitation 30m](runs/velocity_imitation_30m_01/SUMMARY.md) trains the full motor actor on the current velocity plant. [Velocity PPO 01](runs/velocity_hover_ppo_01/SUMMARY.md) and [02](runs/velocity_hover_ppo_02/SUMMARY.md) update encoder, modeled cell parameters and readouts. | Already attempted, including on agile-v5. First PPO overshoots; post-step acceptance then preserves more flight but does not solve regulation. Simply proposing joint training is not a new method. |
| Have we tried explicit local feedback supervision? | [Motor response 01](runs/motor_response_01/SUMMARY.md), [02](runs/motor_response_02/SUMMARY.md), and [wing feedback 01](runs/wing_feedback_01/SUMMARY.md) include paired sensory perturbations and sensor-extension/readout learning. | Older mechanics and checkpoints. Some local gains improve while physical behavior fails. Better probe fitting alone is not a control milestone. |
| Is phase-sensitive imitation the established cause? | [Velocity PPO 10](runs/velocity_hover_ppo_10/SUMMARY.md): removing imitation preserves 30 Hz wingbeats but does not beat the retained parent. First-batch imitation gradient norm is about 0.0252% of physical PPO's. | Current plant, one seed, historical readout. Does not establish imitation phase conflict as the dominant limitation; first-batch gradients are not all subsequent optimizer updates. |
| Can exploration itself break a working flight? | [Frozen exploration test](EXPLORATION_DIAGNOSTIC.md): zero/half/original noise survive 32/32, 30/32, 18/32. [Run 11](runs/velocity_hover_ppo_11/SUMMARY.md) improves velocity error 14.5% and climb 42.7% after halving noise. | Same frozen actor in the diagnostic; actual improvement from a subsequent training trial. Reducing noise is a tested intervention, not a new hypothesis. |
| Can this actor learn corrective control? | [Run 05](runs/velocity_hover_ppo_05/SUMMARY.md) reduces full-flight velocity RMS 24.4→14.92 mm/s. [Run 13](runs/velocity_hover_ppo_13/SUMMARY.md) reduces climb 60.73→13.80 mm and brakes upward motion in six withheld recoveries, but worsens horizontal motion. | Actual full-brain/body behavior. The unresolved task is coordinated regulation with startup retained, not absence of all learning. These are narrow development populations. |
| Have we investigated rewards preferring the wrong tradeoff? | [Run 13](runs/velocity_hover_ppo_13/SUMMARY.md) exact-action reward replay prefers the sideways regression. [Run 14](runs/velocity_hover_ppo_14/SUMMARY.md) fixes the demonstrated ranking, but final cold starts fail while warm recoveries survive. | A real objective mismatch was found and addressed; fixing that ranking alone did not solve learning. Do not restart generic reward tuning without a new counterexample. |
| Does our predictor respond correctly to command changes? | [Residual model 02](world_model/RESIDUAL_02.md): significant velocity effects have 100% sign agreement across tested groups; maximum normalized effect error 4.995%. | Fixed future-command physical branches. This is different from predicting a changed recurrent policy's future commands. |
| What newly failed in full-body model-guided learning? | [Full-body guidance 01](runs/full_body_guidance_01/SUMMARY.md): on one cold-start diagnostic, fixed-neural-history predicted vertical velocity +4.237 mm/s; actual -35.541. Giving the model actually executed commands predicts -35.728. | The optimized future motor histories no longer match the changed actor's live feedback. This specifically implicates the model-based training approximation. It does not invalidate conditional recorded-history PPO replay. |

## Completed work versus plans

The [decoder-authority plan](runs/decoder_authority_01/PLAN.md) proposes small
decoder-parameter perturbations in cold and warm full-brain continuations. The
[run guide](README.md) explicitly records that this was paused before execution
when the user selected the world-model effort. Do not describe its planned
results as measured. The world-model interventions did execute, but held future
commands fixed and therefore answer a different question.

The preferred [shared decoder](FULL_BODY_DECODER.md) is an algebraic consolidation
of run-11's retained function, with upstream tensors unchanged and matched flight
behavior verified. Historical wing-readout experiments inform its ancestry;
they do not authorize reinstating a separate wing head or output mask.

## Consequence for the next decision

Preserve the working plant and single shared 78-output decoder. The accumulated
evidence supports learning difficulty in coordinated, stable closed-loop control;
it does not support restarting from a presumed missing velocity input or calling
the brain unable to learn. The previous encoder-bottleneck suspicion should be
downgraded, although broader representation limitations are not ruled out.

The next bounded intervention should address the demonstrated model-guidance
mismatch: collect fresh full-brain histories, propose small full-decoder changes,
judge them with the actual recurrent brain/body response, and refresh histories
after acceptance. Frequent recollection limits use of stale histories; it does
not mathematically supply the missing within-window neural-feedback derivative.
Use short-horizon proposals and actual closed-loop acceptance, and retain the
analytical comparator. If predictions still fail to rank small live proposals,
stop that approximation and change how recurrent feedback enters learning,
rather than extending optimization on recorded future motor states.

Continuation requires combined velocity/climb improvement with complete cold-start
flights, not improved model loss or one-axis performance. This review launches
none of those experiments and changes no policy or reward.
