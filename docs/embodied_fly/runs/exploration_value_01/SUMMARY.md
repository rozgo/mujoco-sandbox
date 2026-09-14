# Frozen exploration and critic capacity

All16 worlds survive at latent action standard deviations0,.001,.003. At.006,
10of16 fail within ten seconds. Mean episode return at.003 is27.7593 versus
28.0477 at.001 (98.97%). Thus.003 passes the predeclared small-increase gate.
This is a local stochastic probe, not general robustness or an improved actor.

The actor is verified bitwise unchanged.64worlds collect320,000actions in
76.343576s. Setup6.910172s, trace/export preparation12.896729s, total99.209063s.
Two1000-update diagnostic critic fits take1.931850s and1.054376s respectively.
These predict four seconds of measured discounted reward without bootstrap;
they are not PPO value checkpoints. Complete random streams are held out.
Raw-target fit test RMSE1.15456, explained variance.34793; normalized-target
fit1.18085 and.35942. Reward target scaling alone is not a compelling fix.

The saved PPO child's two hidden tanh layers are99.6255% and99.9935% saturated
on these features. Correlation with four-second measured returns is-.16058;
the child estimates another policy/horizon, so this is descriptive only.
Follow-up will examine input standardization and preserve the fixed actor.
The fresh finite-horizon diagnostic critics will not seed the next PPO critic.

All182 tests pass in156.46s; eight focused tests including subsequent fixed-input
calibration pass in1.30s. Full raw trajectories remain ignored outputs with
hashes in diagnostic.json. Prior actor and all failures remain preserved.
