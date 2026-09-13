# Smaller learning steps preserve ground behavior

This repeat changes only Adam learning rate, 1e-5 to 1e-6, against run01. The
parent, seed, curriculum, assistance and all other configuration values match.
Both runs complete exactly151,552transitions and148updates. Run02 takes
181.192194seconds learning; setup9.963105s, collection/inference145.008356s,
optimization36.171425s. Peak CUDA allocation9,441,672,192bytes. These are32
native CPU physics worlds with RTX4090 neural inference/learning.

Unassisted five-second review keeps standing and walking upright with permitted
support. Standing wing RMS is0.05272rad (parent03:0.05704); walking0.04783rad
(parent03:0.05276). Peak deviations0.2441/0.2358rad still fail the full0.2rad
wing gate. Standing loses at most3.04% body height. Tracking gates remain failed,
as does hover. Smaller updates prevent the standing collapse seen in run01 on
this matched development case; they do not solve flight or establish robustness.

All8training failure traces and full capture/model/checkpoint hashes are verified.
Evaluation takes6.726523s setup and32.534879s capture. The resulting checkpoint
is the next curriculum parent for removing hover collection assistance, while
ground_outcome03 remains preserved. No accepted motor release is declared.

Checkpoint SHA-256:
`f5d383485f37478b02b74cf9502c603f35b24e2f08b35e0ca6035b76b0bd87c6`.
Source51f98e5. The full three-command video is being rendered at this checkpoint;
its completed review will be recorded separately.
