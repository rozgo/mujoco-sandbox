# Proposed next experiment: learn to correct established drift

Status: planned after run12, **not implemented or trained in this report**.

Run12 makes vertical control better but worsens horizontal control. The next
hypothesis is that more direct practice from the actor's established errors can
teach their joint correction. An earlier gust experiment already regressed;
this plan therefore uses naturally reached states instead of added forces.

1. Preserve run11 midpoint as the parent and all existing evaluation cases.
   Retain32 worlds,16 CPU physics threads, RTX4090 full-brain inference,1kHz
   physics/500Hz control, .0015 exploration, .005 KL cap, current reward,
   wing-readout-only PPO, critic and light imitation. One shared actor remains.
2. Before training, run the frozen parent normally under zero-velocity commands
   on training starts0–7. Save initialization candidates at2,4,6 seconds only
   where flight is physically valid. Save the complete supported integration
   state plus wing-force bookkeeping, previous action, sensed state, full
   recurrent neural state, and the reward's50-sample history. Record all costs.
3. Implement restoration as an episode initialization only. Body and brain must
   refer to the same preceding history. Rotate each saved reward ring into its
   destination slot's current index; do not silently zero an established history.
   Preserve actual previous actions and matching observations. Keep elapsed
   rollout/termination counters explicit rather than confusing them with saved
   physical time. Do not add state changes during live flight.
4. Verify restored-versus-uninterrupted observation, actor output, reward and
   short physical continuation before updating weights. Check reset isolation
   so restoring one world cannot change another world or its reward history.
   Existing qpos/qvel-only resets are insufficient for this experiment.
5. Allocate16 worlds to existing normal initialization and16 to the saved
   recovery starts. Sample recovery starts from training data only. On reset,
   restore matching body/brain histories; during the episode only the current
   actor drives physics. Same zero-velocity command throughout this hover stage.
6. Train108 rollouts (1,769,472 transitions, expected~10minutes), checking at54
   and108. Time preparation/evaluation separately. Keep the teacher outside
   physical control and preserve every checkpoint, including regressions.
7. Evaluate the same four10-second cold starts. In a separately reported set,
   use withheld parent histories from starts8/9 to test recovery; compare the
   new actor with the frozen parent from the same saved histories. Compare
   first-versus-last one-second averaged horizontal/vertical velocities during
   each complete recovery window, and retain failures in the denominator.

Promotion requires four complete normal flights, total velocity RMS<12.76mm/s,
horizontal RMS<10mm/s and net climb<50mm, with startup dip/error no worse than
the retained parent. Recovery evidence must show reduced measured drift, not
merely completion or a higher training reward. If this block plateaus, do not
queue another unchanged block: pause training for a phase-conditioned diagnostic
of which existing wing actions can brake each measured velocity, and whether
those corrective distinctions survive the encoder/connectome/readout path.
No motor-interface change should be made until that diagnostic identifies a
specific limitation.
