# Sustained flight improves, but walking is forgotten

This is a preserved developmental failure, not a replacement for position_motion01.
The same final checkpoint keeps standing and hover upright for five seconds, but
the walking request produces almost no forward movement. Hover still climbs.

Source cfaf5d2, parent position_motion01. The only planned recipe changes are
five-second episodes instead of two and startup supervision weight 1 instead of
10. Architecture, graph, body and flight reference are unchanged. Training seed
97003; unassisted evaluation seed 97013. This differs from the parent's evaluation
seed, so reported improvements are descriptive rather than a matched-seed effect.

| Measurement | Actual result |
| --- | ---: |
| Setup | 11.640446 s |
| Training | 180.682384 s |
| Collection / forward | 137.172816 s |
| Backward / optimization | 43.480416 s |
| World/action transitions | 172,032 |
| Aggregate simulated experience | 344.064 s |
| Optimizer updates | 168 |
| Peak CUDA allocation | 9,420,270,080 bytes |

32 worlds (11 stand, 11 walk, 10 hover), 16 native CPU physics threads, RTX 4090
neural work. 5 kHz physics / 500 Hz commands. Walking collection executes the
anchored teacher, while standing and hover execute the student. All 262 failed
episodes are hover and their stored traces are verified. There are 22 stand and
22 teacher-walk timeouts, zero complete five-second hover training episodes.

| Command | Stable | Full gate | Result |
| --- | --- | --- | --- |
| Stand | yes | PASS | original form retained; root RMSE 0.233 mm |
| Walk | yes | FAIL | -0.028 cm forward displacement; speed RMSE 1.012 cm/s |
| Hover | yes | FAIL | root RMSE 20.34 mm; reaches 5.43 cm, target 1.87 cm |

Evaluation uses only the frozen student, one checkpoint for every command, no
resets or output substitution. Setup 8.402364 s; capture 29.023983 s; 7,500
transitions / 15 aggregate simulated seconds; zero numerical warnings. The
hover case remains above 1.61 cm and finishes at 4.49 cm. Ground wing/posture
gates pass. All capture hashes, action feedback, physical/graph identities and
frozen normalization/utility parameters are independently checked.

Next trial returns to the walking parent and retains its learned ground commands
during flight learning. It does not start from this stationary walking result.

[Complete three-command review](../../../../previews/embodied_fly/position_sustain_01_all_tasks_v1.mp4).

The 15-second 1600x900 / 50 fps / 1x video was fully decoded (750 frames),
visually inspected across all commands and opened. Render/encode: 59.319069 s.
