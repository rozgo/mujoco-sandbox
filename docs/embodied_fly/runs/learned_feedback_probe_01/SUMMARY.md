# Feedback after sensor-extension learning

Frozen wing_feedback01 is probed on its own recorded histories at 0, 20, 100 and 200 ms. The existing diagnostic recomputes both wing-sensor paths after small angle/speed changes and holds the remaining observations and prior neural state fixed.

Setup 4.439183 s; diagnostic 1.858781 s. There are 75 parallel neural sequences, zero physics transitions and zero training updates. Causal action replay matches within 5.07e-7.

Next-action mean ground angle gains are -0.08183/-0.05271 action/rad for stand/walk, compared with the target -0.66667. Mean speed gains are -0.001276/-0.000276 action per rad/s, compared with -0.005; 4/10 checks respectively have the wrong sign out of 24 per case. Holding inputs for 25 updates gives stronger mean angle gains (-0.21713/-0.17977), but not the requested immediate response.

These are local counterfactual responses on this actor's own trajectories, not a matched causal comparison or a stability proof. Physical evaluation still fails. Keep the complete matrices and acceptance results alongside these summaries.
