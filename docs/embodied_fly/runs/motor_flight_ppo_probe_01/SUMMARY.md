# Direct-flight PPO pilot — not promoted

From angle01, the same 395-input / 78-output graph actor trained from physical
outcomes in **32 native MuJoCo worlds / 16 CPU threads**, with neural learning
on the RTX 4090. Full wing aerodynamics and contacts remain active; no teacher,
wingbeat generator or applied root force drives the rollouts. Initial states
come only from frame zero of the six training-split flight demonstrations.

Measured training wall time **62.569215 s** after **5.463265 s** setup:
35.585713 s collection, 19.551601 s PPO optimization and 7.416602 s explicitly
counted ground rehearsal. Collected **65,536 physical transitions / 13.1072
aggregate simulated seconds**, 16 rollouts, 32 accepted PPO chunk updates and
16,384 ground rehearsal frames. Physics is 20 kHz; the actor controls at 5 kHz.
Actor: 2,410,668 parameters; separate training critic: 235,521. Peak CUDA allocation
18,546,183,680 bytes. Seed 50001, learning rate 5e-6, 128-step rollouts / 32-step
recurrent chunks, two scheduled PPO epochs, 60 ms episode timeouts.

All **892 completed training episodes failed**, lasting 6.2–22 ms. Each failure
trace and the checkpoint/model were hash-verified. Physical-reward gradients are
finite and nonzero in excitability, leak and bias; the actual parameters changed.
This proves the learning path executes, not that it learned flight. Initial PPO
KL excursions stopped later updates in each rollout; the pilot is retained.

Both unassisted 0.3-second flight probes still fell: hover/forward root RMSE
9.901/16.688 mm. All six fixed ground cases remained stable with valid support,
but raw tracking gates failed. Continuous walk and stop stayed stable; resume
toppled. No MuJoCo numerical warnings occurred. Keep angle01 and online01
references and reject promotion. Before more optimization, measure whether wing
commands are decodable from the frozen actor's motor-cell states on excluded
whole expert episodes. That diagnostic will not become a second deployed brain.
