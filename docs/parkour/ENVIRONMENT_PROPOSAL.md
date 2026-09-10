# Proposed visual parkour environments

Research date: 2026-09-10 UTC. This is an engineering proposal, not a reproduction result or a tested training configuration.

## Evidence and reproduction boundary

The [SWAP paper](https://arxiv.org/html/2606.19928v1) describes a 72 kg Apollo, an adversarial motion prior, privileged training signals, and approximately ten hours of training on an RTX 4090. Inspect Figure 2 and Tables I–II for the sensor/network interfaces; Section III-D for rewards and curriculum; Section IV for experiments. These are author-reported results, not estimates for our implementation.

I inspected the [project site](https://swap-parkour.github.io/), sampled the gap, climb and architecture clips, and enumerated its [public repository](https://github.com/swap-parkour/swap-parkour.github.io) at `95da83fee874e2304e53fc603540fcd0ed64de91`. It contains papers and website/media assets. I found no training implementation, checkpoints, robot description or motion-prior dataset there. The project's public organization exposed only this website repository at the time of inspection; that does not establish that no other release exists.

The reference [WMP implementation](https://github.com/bytedance/WMP) at `c232c115ada4517453ebded5019078ba055456de` contains an AMP/PPO trainer and four trot/hop motion files. Its installation instructions depend on old Isaac Gym and Python versions. It is useful for understanding interfaces and potentially retargeting motion data, subject to provenance and license review; it is not the runtime stack proposed here. Its code does not establish SWAP's exact optimizer or missing training settings.

Call our implementation **SWAP-inspired MuJoCo parkour** until architecture, motion prior and training differences have been documented and audited. Keep optimizer choice fixed across the initial architecture comparison. Changing representation, prior, simulator and optimizer simultaneously would make causal claims difficult.

## Robot and physical assumptions

Use the existing Menagerie Go2 as the first supported robot, with replaceable robot configuration. Do not rescale the robot or increase motor limits to match Apollo demonstrations. Define terrain dimensions relative to measured hip spacing and standing height as well as meters; choose the reachable range through experiments.

A local read-only MuJoCo audit found 15.206408 kg total mass and 12 actuators. The front-left calf's first collision cylinder has radius 0.012 m and x-offset 0.008 m, versus 0.013 m and 0.010 m on the front-right. The base inertia's relative reflection discrepancy is approximately 0.002413 in Frobenius norm. Therefore the vendor model is not exactly bilaterally symmetric.

Maintain two explicit model presets: an idealized symmetric benchmark composed without modifying vendor files, and the original physical model for robustness evaluation. Audit all relevant collision shapes, inertias, joint axes and limits, actuators, cameras and reset distributions before claiming exact symmetry. Symmetry of observations should produce appropriately mirrored actions; it does not require both sides to move together on arbitrary terrain.

Preserve actual motor caps, joint travel and full meaningful contact geometry. Calibrate torque-speed and control-delay assumptions before interpreting maximum athletic performance. Measure torque saturation, penetration, landing impulses and energy use. Permit useful foot contact with vertical faces and edges; distinguish it from damaging body impact. A generic rule terminating every non-ground contact would defeat climbing.

## Proposed terrain generator

Each task is a short parameterized lane with a run-up, obstacle and landing/run-out. Generate independent worlds from shared specifications. Reserve a longer connected arena for final evaluation and filming. All numerical ranges below are initial design proposals, not demonstrated robot capabilities.

| Module | Proposed adjustable variables | Initial exploration range | What the experiment measures |
| --- | --- | --- | --- |
| Gap | Width, approach length, landing elevation, edge angle | 0.10–0.50 m gap; expand after successful landings | Anticipation, take-off placement, airborne passage and recovery |
| Platform | Height, top depth, face angle, edge radius | 0.08–0.35 m height initially | Transition from stepping to coordinated climbing |
| Stairs | Rise, tread, number, ascending/descending | 0.04–0.16 m rise initially | Repeated contact and recovery between steps |
| Tilted obstacle pair | Roll of boxes/stairs, signed lateral height difference, skew of gap edges | Mild signed tilt, increasing with measured competence | Transfer between original and mirrored geometry |
| Mixed course | Order, spacing and combinations of individually learned modules | Predetermined unseen combinations | Recovery and skill composition without per-obstacle resets |
| Robustness course | Friction patches, controlled sensor delay/dropout, model asymmetry | Separate sweeps after core tasks work | Limits of the symmetry assumption and reliance on perception |

Use solid boxes and explicitly constructed edges for cliffs and platforms. A gap must be unsupported space; no infinite support plane at walking height may span it. A lower catch floor may terminate a failed episode after contact. Keep collision and sensed surfaces aligned. Overhangs and sparse stepping stones can be later extensions once simpler measurements are reliable.

Each generated case must have a mirror operation transforming geometry, start state, command, material assignment and randomized physical parameters together. Training and evaluation must use different case manifests. For a structural transfer test, train on one signed terrain orientation and reserve the opposite orientation. Keep a separate test of genuinely new shapes and combinations: successful mirroring alone is a narrower result than broad generalization.

## Proposed sensing and learning interface

Provide a calibrated forward depth camera on the sagittal plane. Start at 80×60 pixels, 10 Hz sensing and world-state updates, 50 Hz policy actions and 500 Hz physics; these are our candidate settings. Test sensitivity and benchmark before fixing them. Keep RGB spectator views independent of policy inputs.

The actor interface should expose causal proprioceptive history, current command and learned memory. Use simulation-only state in a separately typed critic/training-target interface. Sensor queues must retain timestamps, validity masks and reset IDs. Never feed exact upcoming terrain, future actions or later sensor frames to the actor. Include the actual executed action history in temporal model inputs.

Train a predictive latent model with observation reconstruction and auxiliary geometric targets. Evaluate its action-conditioned prior over held-out sequences; good reconstruction alone does not demonstrate forecasting. Feed the representation into a learned joint-target policy with torque-bounded physical tracking. Do not reuse the amphibious foot trajectories as live control.

Implement the reflection rules explicitly and test them independently: polar vectors, angular pseudovectors, leg permutations, joint signs, depth pixels, memory, action normalization and stochastic exploration. Verify that applying reflection twice returns the original representation. Check mirrored dynamics numerically with declared tolerances. Paired exploration variances and normalization statistics must respect the same rules.

Treat the motion prior as a first-class configuration and data dependency. First determine the provenance, retargeting quality and physical feasibility of any adopted examples. Save the prior weight, discriminator inputs and data hashes. A prior-free experiment is a distinct ablation, not a silent substitute for a missing dataset.

## Proposed comparisons and success criteria

Compare a conventional recurrent predictive model, the same design with mirrored-data augmentation, and the symmetry-constrained model/policy. Keep observation access, optimizer, task rewards, prior data, training budget and evaluation seeds matched; report parameter counts and compute differences. Add model-only and policy-only symmetry ablations if the initial comparison warrants them.

Advance difficulty using complete traversal plus stable recovery. Track competence by terrain family and retain easier examples. A robot that waits forever, walks around the obstacle or falls across the finish coordinate has not completed the intended maneuver. Progress, take-off, landing and finish events should come from measured state and contact; no timed animation transitions.

Primary outcomes: completion rate by difficulty and held-out suite, with multiple training seeds and uncertainty intervals. Also record gap/height in meters and robot-relative units, fall/impact causes, transit time, saturation, landing stability, energy, prediction error by horizon, and model equivariance error. Report environment steps and wall time separately. Select checkpoint and video cases without consulting the final test set.

## Current stack and feasibility gates

Target current MuJoCo/MuJoCo Warp for batched physics and depth generation, and current PyTorch for learning. MuJoCo Warp now documents batched depth rendering, per-camera resolution, raycasting and heightfield/mesh support in its [official documentation](https://mujoco.readthedocs.io/en/stable/mjwarp/index.html#batch-rendering). This architecture does not require derivatives through contact dynamics. Keep ordinary CPU MuJoCo for macOS viewing and sim-to-sim evaluation.

Use an isolated uv environment for the new learning dependencies. Resolve current releases, test them on the actual GPU, and pin the resulting versions/revisions. Do not inherit obsolete runtime requirements from reference code. Benchmark 64, 256, then larger world counts with sensing, model updates and replay enabled. Watch peak memory and contact capacity; a physics-only throughput number is insufficient. Make a training ETA from that benchmark rather than importing a paper's time.

The first engineering gates are: static scene and depth previews; physical contact/symmetry audit; batched depth plus learning memory benchmark; motion-prior decision; one low obstacle learned and evaluated; architecture comparison; progressively harder courses; final video. Sparse-foothold parkour and autonomous route selection are separate extensions. A supplied velocity command must be labeled as such.

## Proposed video

Show early and later checkpoints on matched cases, then a frozen policy entering an unseen mirrored course. Pair a third-person view with actual depth and a clearly labeled reconstruction or held-out forecast. Do not portray reconstruction as a future prediction or fabricate future robot poses from a model that does not predict them. Finish with matched method comparisons and aggregate success/failure counts. Use a continuous physical traversal for the final course and preserve failed recordings.
