# Sustain altitude correction while preserving standing and walking

Continuation started 2026-09-13 14:33:47 UTC. The prior conversational turn
explained sensor history but changed no authoritative state: no progress under
the goal audit. This turn rechecked both hosts and the latest captures, then
tested reference recovery. No prior training job was running on the GPU host.

Parent position_motion01 walks forward and remains airborne for five seconds,
but climbs to 14.74 cm; the hover target is approximately 2 cm. Standing passes,
walking yaw fails. The unchanged reference recovers 11/12 declared starts,
including 10 cm altitude; the low/fast-falling case touches down and is retained.

Predeclared pilot: continue position_motion01 for 180 seconds, 32 worlds
(11 stand, 11 walk, 10 hover), 16 CPU physics threads and RTX 4090 neural work.
Increase episode duration from two to five seconds and return startup loss
weight from 10 to 1. These are the only recipe changes besides seed/parent.
Use seed 97003, sequence 32, fresh Adam learning rate 0.0001. All motor encoder,
decoder and modeled neuron parameters learn; measured edges and utility stay
fixed. Task weights remain 4:4:1, ground wing weight 10, hover wing weight 2.
Standing and hover execute only the student during collection; walking executes
the anchored training teacher. Standing retains original-form supervision.

The physical body, position actuators, measured-wing force model, 397 causal
inputs, 78 outputs and four graph updates remain unchanged. No runtime teacher,
oscillator, phase input, output masks, flight overrides or extra actor is added.

Review the single final checkpoint for five complete seconds per command using
seed 97013. Preserve every existing gate, report all failures, archive trajectories,
record and open the full 15-second video at 1x. This is a developmental trial;
neither demonstration footage nor training imitation loss proves motor mastery.
Utility, transitions, takeoff/landing and the complete survival goal remain open.
