# Hover PPO with tested exploration and calibrated critic inputs

Active motor-learning goal continuation, September14UTC. Previous goal work
made progress: a frozen64-world experiment established a usable noise increase
and exposed near-total hidden activation saturation in the preceding critic.
Accurate hover and subsequent motor/survival stages remain incomplete.

Parent: assets/embodied_fly/diagnostics/pid_imitation_01_teacher_stage.pt,
SHA315ef3a259ed0e74757163bf58ab22feabbc2049d24992a5028d6802805df1d3.
Preserve the single399-input/78-output MaleCNS actor, graph routing, canonical
wing_position body, instantaneous force law,1000Hz physics and500Hz actions.
No PID, new oscillator, action mask, imitation, utility or ground rehearsal.

Changes from hover_credit_01, declared before training:
- Exploration standard deviation/floor .001 -> .003; all16 frozen probe worlds
  survived and retained98.97% of mean .001 return. .006 caused10/16failures.
- Critic inputs use per-feature mean/std from the first collected rollout,
  frozen afterward, std floor.05 and clip+/-10. Same1713->128->128->1MLP and
  tanh activations; statistics are training-only checkpoint buffers. Fresh
  critic calibration changes its random initial function before its first fit;
  collected GAE targets remain fixed. No trained critic is silently transferred.
- Critic epochs2 ->16 (256updates per2048-action rollout); actor epochs stay2.
  Critic fitting uses detached causal observation/preceding neural state and
  fixed timeout-aware GAE returns. Values remain in physical reward units.
- Actor-frozen warmup4 ->2rollouts; each now has8x the value fitting updates.
  Warmup, calibration and extra critic fitting count inside training time.
- Requested training600 ->300seconds, complete the current rollout/update.
  Seed120601. This is a development trial, not a matched causal benchmark.

Keep64hoverworlds,16CPU MuJoCo/mjbatch threads,RTX4090 neural training,
horizon2048, recurrent sequence128, ten-second episodes, actorLR1e-7,
criticLR1e-4, KL limit.03, entropy0, gamma.9996000799893344,
lambda.9994001799640054. Same bounded reward including20mm/s vertical-speed
scale and near-nominal reset curriculum. Fresh actor/critic Adam from imitation.

Evaluate the same three ten-second learned starts and independent accepted PID.
Progress requires all3airborne, nominal position RMS and altitude RMS both
improved>=10%, neither perturbed case worse>10%, settled wing frequency25–35Hz
and repeated height ripple<.2mm. Original stricter accurate-hover gates stay.
Retain all failures. Render/open PID/PPO and parent/PPO comparisons. If this
fails, preserve the preferred imitation parent; do not promote on survival alone.
