# Physics-integrated residual follow-up

The user requested a measured two-minute trial after reviewing the first pilot.
Keep the live plant, actor and MaleCNS untouched. This learns a predictor only.

- Fresh causal history/action/GRU latent model: 60 seconds of actual optimizer
  steps on RTX 4090, batch 128. Recorded 500 Hz states/actions; 200 ms horizon.
- Add matched positive/negative wing-command interventions from training and
  validation episodes only. Preserve test episodes 8/9 and previous test cases.
- Freeze the latent model. A latent-only prober outputs six bounded acceleration
  residuals: world linear (100 cm/s^2 scale) and body angular (20 rad/s^2).
  No learned direct future-state correction or initial-state MLP bypass.
- Known capped wing PD, measured-wing force law, drag/gravity and locked-inertia
  body dynamics integrate at 1 kHz. Residuals are held across the two ticks of
  each 500 Hz command. Loss backpropagates through the full physical integration.
  Wings need no learned dynamics correction: they are independent coordinates.
- Prober training: 60 seconds of optimizer updates, batch 128. Equal shares of
  recorded-flight windows and paired unmodified/perturbed continuations. Mean
  squared normalized position, velocity, rotation and angular-velocity error;
  0.1-weight paired velocity/angular-effect loss; 1e-4 residual regularization.
  Zero residual is a saved baseline; choose the lowest validation velocity error.
- Report data collection, precomputation, graph setup, optimizer and validation
  times separately. CUDA graphs may replay the identical differentiable training
  computation to remove Python overhead; no updates during setup/warmup.
- Validate zero-residual agreement and finite-difference gradients before
  learning. Apply the unchanged forecast and intervention gates from PLAN.md.
  Compare learned versus zero residual to distinguish physics-prior performance
  from actual learned improvement. Preserve all unsuccessful trials/results.

Initial analytical limitations remain: locked initial articulation, approximate
joint stops, no contact prediction. A successful predictor is not yet a better
fly policy. Report a concrete next step after the held-out evaluation.
