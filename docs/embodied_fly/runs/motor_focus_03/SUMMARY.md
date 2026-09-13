# Learning quiet wings on the ground

The [matched actuator interventions](../motor_interventions_01/SUMMARY.md) showed
that quieting the six wings rescues short ground posture. This pilot teaches that
behavior through the same actor rather than applying a runtime override. It adds
2× wing-channel MSE to ground tasks, matching the existing hover wing weight.
The body, force law,78 outputs,397 inputs, fixed measured graph and32-world split
are unchanged. This is online imitation, not PPO or a recovered biological brain.

Source57b0f35, seed71003, motor-focus02 parent, fresh Adam1e-5,32-step recurrent
chunks and25% reference assistance. Actual training took **181.184887 s**, setup
7.538387 s, collection/forward141.558038 s and backward/optimization39.622251 s.
**160 updates /163,840 physical transitions /327.680 aggregate simulated seconds**.
RTX4090 graph/learning; native CPU MuJoCo/mjbatch physics,16 threads,5 kHz physics,
500 Hz control. Peak CUDA allocation9,411,467,264 bytes. All parameters outside
utility/context heads remain trainable; those inactive heads stay bitwise fixed.

Of160 completed assisted episodes,152 reached two seconds and8 fell. Per task:
47/55 stand timeouts,55/55 walk and50/50 hover. These assisted counts do not imply
student hover. All8 causal failure traces are retained and verified.

**Unassisted two-second standing and walking stay upright with valid foot support.**
Both previously fell. Their minimum upright values exceed0.984; prohibited support
is zero. Ground wing velocity RMS over the matched first200 ms declines from
5.71→3.54 rad/s standing and4.95→2.93 walking. There was also additional training,
so this is not a matched no-new-loss control experiment. Heading and speed gates
still fail; hover falls. The [complete three-task review](../../../../previews/embodied_fly/motor_focus_03_all_tasks_v1.mp4)
keeps all outcomes visible, with actual simulated neural state and observer eyes.

A longer native evaluator, using the same physical fingerprint and another
predeclared initial heading (seed73001), confirms **five seconds of upright
standing**, no prohibited support or numerical warnings. It drifts0.801 mm and
still fails raw heading tracking. Walking reaches about3.28 s before losing
posture. The [complete ten-second ground review](../../../../previews/embodied_fly/motor_focus_03_ground5s_v1.mp4)
includes that fall. Both videos run at1× and were decoded, inspected and opened.

This is a useful ground-stability checkpoint, not accepted locomotion or flight.
The earlier reviewed walking and every failed candidate remain preserved. Further
work must retain this ground control while improving longer walking and unassisted
wing startup. Utility selection and survival decisions remain deferred.
