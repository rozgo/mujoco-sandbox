# Feedback probe on the nonlinear online actor

Frozen checkpoint nonlinear_online01 is replayed on its own canonical physical histories at0,20,100and200ms. Small independent changes to each wing angle and speed recompute both measured input paths while preserving other observations and prior neural state. Responses are observed after1,5and25held-input updates. These are sensory counterfactuals, not physical rollouts.

Setup5.506291s;diagnostic1.875014s.75parallel neural sequences;zero physical transitions andtrainingupdates. Original-history replay agrees within4.33e-7.

For the24standing angle checks at the next action, mean restoring gain is-0.07152action/rad against a ground-target gain of-0.66667. Walking mean is-0.05204. Standing/walking speed gains average-0.000717/-0.000405action perrad/s against-0.005;7and8checks respectively have the wrong sign. Holding observations for25updates increases angle-response magnitude, but it remains too weak on average. This does not prove a unique causal origin or establish closed-loop stability.

The decoder-only network has weak or incorrect local feedback on these real states. The next declared pilot trains the existing measured-wing sensor extension together with the nonlinear decoder through the fixed graph. Original graph topology, cell parameters, motor-decoder base and physics stay fixed. Additional ground body-action distillation counters changes to non-wing behavior; it is a training loss, not an output override.
