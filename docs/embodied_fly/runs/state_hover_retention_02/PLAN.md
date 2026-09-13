# Smaller-step repeat of the retained ground / state-hover pilot

The frozen-reset audit reproduces the failure with the training batch size and
identical initial inputs. The next pilot changes **only Adam learning rate from
1e-5 to 1e-6**, starting again from ground_outcome_03, not from the failed actor.
This tests whether smaller updates preserve the ground controller while learning
the new hover labels. No inference, body, sensor or force-law changes.

Retain run01 settings: seed93001; 32worlds (11stand/11walk/10hover); 16CPU threads;
32-step chunks; fresh Adam; 180-second learning allowance; 2-second episodes;
ground actor-only collection, hover80%teacher/20%student; task-loss weights4:4:1;
ground wing weight10, hover wing weight2; same state-hover reference gains;
independent frozen parent reference. One resulting checkpoint, no runtime teacher.

Evaluate all three commands unassisted for five seconds with seed72001; preserve
and open the full video. Training timing and actual transitions are measured,
not assumed equal. The parent remains selected unless the complete result improves.
