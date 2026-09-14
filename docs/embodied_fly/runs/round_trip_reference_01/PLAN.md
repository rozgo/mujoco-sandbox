# Six closed flight paths on the accepted plant

User-proposed curriculum: left/right, right/left, forward/backward,
backward/forward, up/down, down/up, each returning to its starting position.
This first check uses the accepted PID reference only. No learned actor or new
training is present. Preserve the preferred imitation checkpoint and all demos.

Seven worlds: the six sequences plus stationary hover. Same canonical physical
body, zero-mass/collision-free wings, per-tick wing-force law, 1,000 Hz physics,
500 Hz actions. The compiled model hash must match the recorded accepted model.
Wing control is the only active force path; live body poses are never assigned.

Targets are +/-1.5 mm relative to each initial position, along world x/y/z at
canonical heading. Hold start 0–1s; smooth quintic ramp to the first target 1–2s;
hold 2–4s; ramp to the opposite target 4–5s; hold 5–7s; ramp home 7–8s; hold home
8–12s. Do not confuse prescribed targets with prescribed body movement.

Predeclare success: no physical failure; maximum position error below 0.5 mm in
each target's final 0.5-second hold and the final returned second; final-hold RMS
speed below 1.5 mm/s. Preserve all cases/failures. This tests directional control
authority of the plant/reference, not a trained brain, flight generalization or
new motor-policy acceptance. Random distances/timing and stationary rehearsal
belong to the subsequent proposed learned curriculum.
