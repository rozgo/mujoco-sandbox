# Isolating ground motor instability

This is a **diagnostic**, not accepted policy behavior. Source076ed79 runs the
motor-focus02 checkpoint in14 matched one-second cases: stand and1 cm/s walk,
with seven actuator substitutions each. Same canonical physical body, no
learning, no live pose writes or resets, and actual executed-action feedback.

Full-graph CUDA inference and native CPU physics took8.442436 s, setup6.907916 s.
The14 cases provide7,000 physical transitions /14 aggregate sim seconds. All
complete captures are finite, hash-verified and causally logged; no MuJoCo warnings.

| Reference-controlled outputs | Stand upright for1 s | Walk upright for1 s | Walk speed RMSE (cm/s) |
| --- | --- | --- | --- |
| None: full student | No; exits0.388 s | No; exits0.458 s | 0.950 |
| Six wings | Yes | Yes | 0.961 |
| Nineteen previously inactive channels | Yes | Yes | 0.936 |
| Six foot-adhesion outputs | Yes | Yes | 0.850 |
| Forty-eight leg joints | Yes | Yes | 0.509 |
| Leg joints and foot adhesion | Yes | Yes | 0.485 |
| All78 outputs | Yes | Yes | 0.513 |

Every substituted case had zero prohibited-support load. Quiet wing commands
alone rescue short ground posture, but do not recover walking speed. Adhesion
and leg substitutions also stabilize the body, so wings are not proven to be the
sole cause. The actionable hypothesis is to teach quiet ground-wing control more
strongly before broadening the motor curriculum. No runtime output override or
force gate is adopted from these diagnostics.
