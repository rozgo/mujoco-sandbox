# Continue physical PPO with measured, smaller exploration

The frozen pilot04 policy fails3/8 sampled five-second nominal starts at .003
latent standard deviation, versus0/8 at .001. Deterministic controls remain
airborne in both captures. Smaller noise gives sampled height spans5.98–6.46 mm
instead of5.23–10.00 mm among prior survivors; failed paths remain reported
separately. This motivates reducing perturbations while learning accurate hold.
It is not a claim of general robustness or a new learned capability.

Continue pilot04's actor, critic and both Adam histories for a requested600
seconds,64 hover worlds,16 CPU physics threads,RTX4090 neural work. Keep the
accepted body,1kHz physics,500Hz actions,78 outputs,399 inputs,fixed graph,
bounded physical rewards,512-action rollouts,128-action recurrent gradients,
two epochs,gamma .999,GAE .995,KL .03,zero entropy and five-second episodes.

Explicit changes: reset only the exploration parameter and its Adam moments to
std/floor .001; reduce actor LR3e-7 ->1e-7 because the narrower distribution is
more sensitive to a mean-action change. These two changes are paired design
choices, not an isolated learning-rate experiment. Critic LR stays1e-4. Resume
critic/critic Adam; do not repeat the completed critic warmup. Seed120205.
No PID, imitation, phase clock, output mask or ground rehearsal is introduced.

Freeze and evaluate the final checkpoint on the same three ten-second starts
plus the accepted nominal PID. Preserve earlier candidates and strict gates.
Render/open another matched1x PID/PPO video even if the new candidate fails.
Report training time,experience,KL and failure rates separately from physical
acceptance. Do not advance to the next motor skill solely on training return.
