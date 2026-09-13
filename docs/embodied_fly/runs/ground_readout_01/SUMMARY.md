# Frozen wing-output fit (01)

**Not selected.** Fitting six existing wing output rows reduces held-out command
MSE by 72.9%, but autonomous standing falls and walking wing posture worsens.
Keep motor06 preferred. Offline fit quality is not physical control success.

Parent06 executes every action during collection, with no teacher assistance or
resets. All failures remain in the saved histories. Counterfactual wing angle
and velocity observations use copied actual prior neural state; they are not
additional simulated experience. The inputs to the fitted output layer are the
existing 256 motor hidden features after MaleCNS activity.

- Source4fbd5db; collectionseed73009;32worlds/16native CPU MuJoCo threads.
- 5kHz physics/500Hz control;RTX4090 for frozen brain inference and ridge fitting.
- Collection31.150358s plus5.250982s setup,32,000physicaltransitions/64aggregate simulated seconds.
- 56,000featureexamples,52,800synthetic; split by complete world, including every variant.
- Fit0.170996s plus1.533296s setup; five declared ridge penalties, selected0.001.
- Held-out balanced action MSE0.00318306→0.00086160. All upstream parameters and
  non-wing rows are bitwise unchanged;1,542existing parameters fitted,zero new ones.
- Same motor-only context and physical fingerprint; no runtime mask or helper.

Five-second evaluation at developmentseed72001 takes31.768500s plus5.218884s
setup. Standing falls; walking remains upright with permitted support but wing
RMS grows0.0631→0.2298rad. Hover falls. Every complete gate fails,zero numerical
warnings. Hashes, finite captures, bounded actions and causal feedback verified.

[Complete review](../../../../previews/embodied_fly/ground_readout_01_all_tasks_v1.mp4):
15s,750frames,1600×900,50fps,1×. Fully decoded, sampled frames inspected and
automatically opened on the Mac; every failed case is included.

Next inspect sensitivity to previous wing commands while holding actual joint
measurements fixed. Ground wing channels were not clipped in parent06's standing
capture. Neither fact proves a complete instability mechanism; retain measured
evidence before another training change.
