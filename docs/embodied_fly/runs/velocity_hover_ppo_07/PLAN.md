# Vertical regulation reward adjustment

The unchanged ten-minute run 06 preserved four ten-second flights but plateaued:
net climb about 101 mm versus 106 mm in run 05; horizontal RMS worsened from
9.3 to 15.4 mm/s. Preserve run 06 and return to run 05 final, SHA256
`3106cb363aa75dab78322181ad9f46c229a8a29ca8f1f0599bc18e8d59763a32`.

Apply the already-agreed conditional adjustment: vertical tracking reward rate
**2 -> 3**, retaining its 0.5 cm/s width and 100 ms causal reward-only average.
All other reward terms, failure rules, body, force law, sensor/action interfaces,
command, clocks, optimizer settings and trainable scope remain unchanged.
The maximum live reward rate becomes 6 instead of 5. No height target or direct
force correction is added. Restore the parent's critic/Adam/input calibration,
and use the first rollout for critic updates only on the revised reward.

Same 32 worlds and 16 CPU MuJoCo/mjbatch threads; RTX 4090 runs neural computation.
1,000 Hz physics, 500 Hz actor; same fixed MaleCNS graph, 391 observations,
78 actions, four zero velocity commands. Only the existing 105,222-parameter
wing readout is trainable. PPO horizon 512, sequence 64, two actor epochs,
four critic epochs, proposed actor LR 3e-6, KL ceiling .02 with rollback,
fixed .003 exploration, same light PID-label anchor. Retain exact live neural
memory through head updates. No PID actuator assistance or dynamics changes.

Run one short discarded smoke to verify the explicit reward transition,
first-rollout actor freeze, retained critic calibration and subsequent updates.
Then start from run 05 again for **600 measured training seconds**, including
critic adaptation and finishing the current rollout. Do not count smoke weights
as training ancestry. Setup, replay audit, evaluation, checkpoint IO and rendering
remain separate. Evaluate the same four starts at halfway and completion.

Success remains four ten-second flights, climb toward 50 mm or less per case,
and sideways RMS near or below the parent's 9.3 mm/s. More than 10 mm/s sideways
is a material regression. Preserve and report both midpoint and final; final
is the primary result. Render, fully decode, inspect and open the 1x comparison.
