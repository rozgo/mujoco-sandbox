# Corrective flight warm start: one-minute pilot

This checkpoint does not establish learned flight and does not replace online01.
Both teacher-free airborne probes fell. Five of six fixed ground cases retained
stable, permitted support; normal walking fell. The continuous walk and stop
phases remained stable, but resuming walking toppled. All raw task gates and
failure captures are retained. There were no numerical warnings.

Eight training episodes used 15% student and 85% inherited-teacher actuator
commands, with the teacher providing correction labels at each actual visited
state. All eight stayed inside the original flight demonstration envelope.
Collection took **45.966942 seconds**, after **5.629544 seconds** of setup, for
**12,000 transitions / 2.4 simulated seconds**. The teacher, student and executed
commands are separate in the data. The student sees the previous executed action;
only the teacher has a reference trajectory and supplied wingbeat generator.
This assistance is training-only and is not counted as student success.

The pilot resumed `motor_flight_probe_02` with its Adam state. It used original
flight demonstrations, corrected flight demonstrations and retained online01
ground episodes. Ground loss had multiplier 4; clocks were sampled equally.
The recurrence used 64 burn-in and 32 supervised steps, with 25% reset-start
batches. At 5 kHz, the 6.4 ms supervised window exceeds one nominal wingbeat.
This combined change cannot isolate the effect of correction data, history or
the new retention weight separately.

| Measurement | Result |
| --- | --- |
| Device | RTX 4090 |
| Physics for collection/evaluation | Native MuJoCo CPU |
| Offline neural sequences | 32 |
| Optimization wall time | 60.676444 s |
| Setup / validation | 5.831953 s / 1.593451 s |
| Updates / supervised examples | 82 / 83,968 |
| Unique training frames | 26,000 |
| Ground / flight updates | 41 / 41 |
| Peak CUDA allocation | 9,586,374,656 bytes |
| Ground validation MSE, before → after | 0.002721 → 0.001536 |
| Flight validation MSE, before → after | 0.058950 → 0.047284 |

The validation data and context differ from earlier pilots, so their absolute
MSEs are not a matched between-run comparison. Physical tests use the same fixed
development cases. Hover root-position RMSE was 9.095 mm over 0.3 s; forward
flight at 10 cm/s was 15.951 mm. Both lost altitude. The stable stop had 0.0246 mm
late drift, but its subsequent failed resume prevents accepting the transition.

Training source `b57a1c4`, seed 47001, learning rate 3e-5. Same shared 389-input,
78-output, 166,700-neuron graph actor; no extra deployed controller.
Checkpoint SHA-256:
`c66ff3faecaad38c9323238e7ef97bb6d83e892df900aef5d6176d5e1fd386db`.
Parent SHA-256:
`ab0c4f53ee43579483c026c017d8cd1c9103dbfe12f6da96a8771d2ae4a86503`.

Next address the independently measured wing-angle clipping, preserving old
checkpoint reproduction and testing a neutral observation extension before
learning. The full flight, takeoff/landing, survival and multi-agent goal remains
incomplete.
