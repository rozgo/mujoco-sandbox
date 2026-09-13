# Rejected sweep candidate: full diagnostic

This exports the search's best unconstrained short-hover candidate, gain1.3/bias0.02, into the same two existing output rows. It is not selected because ground retention failed. All other state and all physical parameters match retention02 exactly.

A fresh five-second-per-command evaluation (seed72003) takes7.357s setup and31.905s capture. Stand/walk remain upright on permitted supports; standing wing RMS worsens to0.1541rad, walking wing RMS is0.05983rad. Hover falls and root RMS is25.544mm. All strict task gates fail, with zero numerical warnings. Ground, hover and failures are all preserved.

The [full15s film](../../../../previews/embodied_fly/sweep_probe_01_all_tasks_v1.mp4) is decoded and visually inspected, and was opened automatically.750frames,1600×900,50fps,1×. Rendering/encoding60.404s. The full trajectories, bounded causal actions, model and checkpoint hashes and exact weight transform were independently verified.
