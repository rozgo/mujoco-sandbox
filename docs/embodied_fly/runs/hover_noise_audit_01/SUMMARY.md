# Existing exploration can interrupt otherwise sustained flight

The frozen pilot04 actor runs in 16 independent worlds from the same nominal
start. Eight execute its deterministic mean; eight sample the saved .003 latent
standard deviation. All mean replicas stay airborne for 5 s. Three sampled paths
fail at 1.132, 1.296 and 1.310 s. Remaining sampled height spans range 5.235–10.002 mm,
versus 5.566 mm for mean control. Mean lift in the sampled survivors is 1.0113 body
weights, versus 1.0104 for deterministic control. Added noise does not simply
supply missing mean lift; it destabilizes some of these repeated-start paths.

The full-graph forward likelihood replay has mean approximate KL 1.0933e-5,
maximum individual log-probability difference .19263, without optimizer updates.
It is not bitwise identical, but mean divergence is much smaller than the .03
PPO stop threshold. This checks 128 forward steps with gradient recording and
detached recurrent state between them; it does not validate long-sequence gradients.

Setup 6.721828 s, physical capture 22.267412 s, replay .817538 s; 40,000 physical
transitions and zero training updates. Source 4b1a860. Failed motion continues
without resets and is retained in the hashed capture. Deterministic replicas
are not independent generalization cases. See [raw results](evaluation.json),
[summary statistics](SUMMARY.json) and [predeclared plan](PLAN.md).
