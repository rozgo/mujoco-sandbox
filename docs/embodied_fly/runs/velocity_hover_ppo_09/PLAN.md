# Learn velocity recovery through varied physical experience

Continue the active flight-improvement goal. Run 08's vector reward correction
did not improve flight: four ten-second flights, but climb 135 mm and total RMS
16.43 mm/s. Keep run 05 as parent (SHA256
`3106cb363aa75dab78322181ad9f46c229a8a29ca8f1f0599bc18e8d59763a32`).

Two frozen-weight diagnostics constrain the next change. Matching the sensor
timing used by PPO collection changes run 05's climb by only 0.23 mm; it is not
the main explanation. Counterfactual measured-velocity inputs reach the motor
features in all three independent directions. The encoder is not missing those
signals. Wing actions respond less strongly to vertical input, though differing
plant sensitivities mean this alone does not establish a decoder defect.

Keep the vector reward and all PPO/network settings from run 08. Change only
the training experience relative to that trial: **8 calm worlds + 24 worlds
with brief physical force pulses**. All commands remain zero. Every pulse chooses
a random world axis and sign, lasts 0.35 s, and is separated by a random 1–2.5 s
quiet interval. Its force equals total mass times a 1–2 cm/s equivalent drift
divided by the existing axis drag timescale. This is at most approximately
2.55% of body weight vertically and 1.70% horizontally. Apply at the thorax COM,
alongside the unchanged wing wrench. No pose/velocity writes, actuator assistance,
new motor outputs, or schedule information supplied to actor/critic.

The existing drag timescales (0.08 s vertical, 0.12 s horizontal) would erase
reset-only offsets quickly; pulses give the actor recovery experience after its
wing pattern is established. The sampler has its own declared seed, restarts
with new training blocks and is recorded with every event. Reset clears only
that world's pulse state after native reset clears applied force. Calm-world
preservation, force composition, no accumulation and pulse removal are tested.

This trial starts from run 05 again for a controlled comparison with run 08;
it does not inherit the regressed run-08 weights. Restore actor Adam, critic/Adam
and sampling RNG; one critic-only rollout adapts to the vector objective. Keep
391 observations, 78 actions, the fixed MaleCNS graph, same FlyBody/v5 force law,
1 kHz physics, 500 Hz actor, 32 worlds and 16 CPU physics threads. RTX 4090 handles
neural computation. Same 105,222-parameter wing readout, frozen upstream actor,
512-step rollouts, 64-step sequences, LR 3e-6, KL .02, .003 exploration, two actor
epochs, four critic epochs and light PID-label anchor.

Discard a short smoke after verifying actual pulses and all invariants. Then
600 measured training seconds, halfway/final evaluation on the original four
starts **without disturbances**. All goal criteria remain: four ten-second
flights, lower total RMS than run 05, climb at most 50 mm and horizontal RMS at
most 10 mm/s in every case. Continue work if the trial fails. Record/open the
comparison and retain every tested checkpoint and measured cost.
