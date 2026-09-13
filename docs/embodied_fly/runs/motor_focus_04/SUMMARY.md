# Initial-form imitation, assisted continuation (04)

**Not accepted.** The new reference can hold the initial body and wings, but
this assisted imitation checkpoint regresses autonomous ground control.

Same physical model, MaleCNS actor and three-command contract as 03. Stand now
targets all initial position-actuated joints; both ground tasks use restoring
wing torques from measured angle/velocity. Ground wing loss weight 2→10.

- 180.182772 s training; 7.528581 s setup.
- 32 worlds(11 stand/11 walk/10 hover), 16 native CPU MuJoCo threads, RTX 4090 brain.
- 161 updates/164, 864 actual transitions/329.728 aggregate simulated seconds.
- Collection/forward 140.225135 s; backward 39.940857 s.
- 9, 411, 467, 264 bytes peak CUDA allocation.
- 25% reference actuation; 149/160 completed assisted episodes reach 2 s, 11 fall.
- All 3 unassisted 5-second cases fail; stand/walk first exit at 0.344/0.338 s.
- All captured states, executed-action feedback, model and checkpoint hashes
  verified; failure traces retained. No runtime masks or physical pose writes.

The early wing diagnosis records a 0.155-magnitude erroneous command at rest.
At 100 ms all six standing wing commands reinforce displacement. Assistance
keeps the training distribution near the reference, which does not establish
autonomous correction. Next change only execution mixture 25%→0%, keeping
corrective labels and the same actor/body/objective.

[Complete failed review](../../../../previews/embodied_fly/motor_focus_04_all_tasks_v1.mp4):
15 s, 750 frames, 1600×900, 50 fps, 1×. Full decode/visual QA passed; automatically
opened on the Mac. Render/encode 59.501659 s. All 3 cases remain included.
