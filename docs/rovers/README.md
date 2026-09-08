# RC rover communications demo

Six suspended RC vehicles survey a 20 × 16 m inspection yard in MuJoCo. Three scouts discover eight yellow inspection markers; three inspectors travel to locally known markers and command a one-second inspection stop while the marker is visible. Observations, claims and completion records travel over a packet-level LoRa model. Radio delivery can change another rover's route before it sees the marker itself.

[Watch the complete degraded Reticulum run](../../previews/rovers/reticulum_degraded.mp4) · [Dashboard preview](../../previews/rovers/reticulum_degraded.png) · [Vehicle detail](../../previews/rovers/detail.png)

The video covers all 150 simulated seconds at 2× speed, then holds the last frame for three seconds. It includes the third-person overview, all six front cameras, packet delivery/loss trails, radio status, per-rover knowledge counts and a selected rover's marker beliefs. Camera pixels are observer output and never consume LoRa bandwidth.

## Run on macOS

From the repository root:

```sh
uv sync --locked --extra reticulum

# Live demo, physical movement starts automatically:
uv run --extra reticulum rover-comms view --reticulum --case degraded

# Lightweight version without the optional Reticulum stack:
uv run rover-comms view

# Frozen scene and all static preview images:
uv run rover-comms view --static
uv run rover-comms preview

# Save a full physical run, telemetry, packet events and metrics:
uv run --extra reticulum rover-comms run --reticulum --case degraded --output outputs/rovers/my-run

# Record that exact saved run without repeating the simulation:
uv run rover-comms record --run-dir outputs/rovers/my-run --speed 2 --output previews/rovers/my-run.mp4

# Paired healthy/degraded/disabled experiments with three radio seeds:
uv run --extra reticulum rover-comms compare --reticulum --seeds 3 --save-runs --output outputs/rovers/reticulum-comparison
uv run rover-comms compare --seeds 3 --output outputs/rovers/direct-comparison

# Includes full physical missions and an optional real-stack round trip:
uv run --extra reticulum pytest -q
```

Controls: **1–6** select R0–R5 front cameras; **0/O** overview; **T** overhead; **F** follows the selected rover with mouse orbit/zoom; **Space** pauses/resumes. `--speed 2` doubles simulated time per wall second. The viewer freezes at the configured `--duration` (150 seconds by default); close and relaunch to restart. macOS automatically uses `mjpython`, with the same uv Python-library lookup fix as the hexapod demo. Tested on Apple Silicon macOS, Python 3.12.12, MuJoCo 3.12.0 and Reticulum 1.5.2. Linux and Intel Mac were not tested.

## Physics and sensing

The rovers move through tire contact forces. Controllers set only 24 wheel velocity actuators and 12 front steering position actuators. There are no prescribed chassis poses, external pushing forces or kinematic motion in a live run. Assigning recorded poses is used only to render a previously simulated trajectory.

| Property | Choice |
| --- | --- |
| Total mass per rover | 4.18 kg |
| Chassis / battery / sensor housing | 2.20 / 0.60 / 0.12 kg |
| Antenna and lights | 0.04 kg total |
| Four suspension carriers | 0.08 kg each |
| Two steering carriers | 0.03 kg each |
| Four tires, hubs and spokes | 0.21 kg per wheel |
| Wheel radius / wheelbase / track | 0.09 / 0.38 / 0.48 m |
| Front steering joint limits | ±0.65 rad (±37.2°) |
| Steering actuator | 35 Nm/rad proportional gain, 2 Nms/rad damping, ±2 Nm torque |
| Drive joints | Continuous rotation; command limit ±12 rad/s; torque limit ±1.2 Nm |
| Drive velocity gain | 0.35 Nms/rad |
| Passive suspension | ±0.025 m travel; 1700 N/m spring; 24 Ns/m damping |
| Tire friction | Sliding 1.1, torsional 0.01 m, rolling 0.001 m; four-dimensional contact |
| Motion policy | Ackermann steering, differential inner/outer wheel rates, forward command ≤0.65 m/s |
| Physics / policy rate | 250 Hz implicit-fast integration / 10 Hz control |

The wheel track was widened after steering tests found tire/chassis interference. A full ±0.65 rad steering sweep now clears the chassis. A local pass-right detour and reverse recovery handle nearby vehicles; inspection recovery backs up if a marker passes behind the nose camera. Mild vehicle bumps remain: the largest inter-vehicle penetration in the 18 validation runs was 5.6 mm. All runs remained upright and finite, with zero MuJoCo warnings and actuator forces within their limits. This is a demonstration controller, not a collision-free fleet planner. The shallow ramp is physical, but public mission routes avoid it.

Each agent receives its own ideal odometry; a synthetic marker detector with a 2.5 m range, ±65° horizontal field of view and MuJoCo ray-based occlusion; and anonymous 360° peer proximity positions within 2 m. The proximity sensor has no occlusion model. Marker recognition and odometry have no measurement noise. Images are real scene camera renders, but detection is geometric rather than computer vision.

Agents share a public static building map and predetermined survey routes. The map includes no marker database. `agent.py` has no MuJoCo access: it receives immutable local observations and bytes actually delivered to that receiver. A received marker belief records the observation time, sender, packet ID and delivery time. Newer evidence replaces older evidence. Lost or queued messages cannot change another rover's belief. Completed records may be known even when the marker position is not known.

The policy uses nearest-known-target selection, time-limited claims, local distance-based claim arbitration, application message deduplication, one-hop scout relaying and periodic completion gossip. It continues patrolling through radio outages. It is deliberately simple and can make suboptimal route choices.

## Radio and optional real Reticulum

MuJoCo supplies position, geometry, motion and rendering. `radio.py` supplies a separate discrete packet channel. This is not an RF waveform simulation or a calibrated range prediction.

| Parameter | Assumed value |
| --- | --- |
| LoRa modulation | SF7, 125 kHz bandwidth, coding rate 4/5 |
| Packet format | Explicit header, CRC, eight-symbol preamble |
| Transmit power / sensitivity | 14 dBm / −123 dBm |
| Background noise / required SNR | −112 dBm / −7.5 dB |
| Log-distance loss | 31.7 dB at 1 m, exponent 2.7; distance clamped to ≥1 m |
| Building attenuation | 12 dB for each intersected horizontal footprint |
| Capture approximation | Desired packet must exceed each interferer by 6 dB |
| Queue / access | 16 queued packets per rover; random backoff; approximate local carrier sensing; half duplex |
| Airtime budget | 0.08 seconds per simulated second, maximum 0.2-second burst credit |
| Residual loss | Seeded 2% healthy loss, in addition to modeled interference and outages |

The airtime budget is an engineering constraint, not a regional regulatory limit. The path-loss constant is a nominal sub-GHz assumption; there is no frequency-dependent propagation model. Building loss uses 2D footprints, without diffraction or multipath. Carrier sensing and capture are approximations, with one shared channel and one modulation profile. The model does not implement LoRaWAN, retransmission acknowledgements or real hardware.

A 14-byte application message contains kind, marker, origin, relay TTL, sequence, observation milliseconds and centimeter-quantized X/Y. `seen` and `done` carry marker coordinates; `claim` carries the claimant's own observed position. A direct message uses 46.336 ms airtime. Reticulum's emitted frame is 33 bytes for this message and uses 71.936 ms. Both charge the actual number of on-air bytes. Receiver outcomes accumulate across the full airtime, then become deliverable on the next 4 ms simulation tick. The reported mean delivery latency measures airtime plus tick quantization; it excludes queue waiting and the subsequent 10 Hz agent update.

`--reticulum` runs six genuine Reticulum stack instances in separate local processes. Each uses an isolated custom simulated interface, with shared-instance connections, transport routing and external interfaces disabled. The parent simulation submits emitted raw frames to the same LoRa channel and feeds only successfully delivered frames to the receiving stack. Real Reticulum decoding invokes the endpoint callback before the application receives the payload. All overhead is charged to the channel.

This exercises **Reticulum PLAIN broadcast framing and delivery**. It does not exercise encrypted sessions, identity discovery, Reticulum multi-hop routing, RNode hardware or Internet links. Relaying is the explicit application TTL policy. The integration uses Reticulum's internal synchronous inbound option, is tested with 1.5.2 and is constrained to the 1.5 series in `pyproject.toml`; `uv.lock` pins the tested release.

The degraded case turns R0's radio off at 8–28 s and R3's at 36–64 s. From 18–76 s it raises receiver noise to −54 dBm within a 7 m radius of the yard center, −74 dBm outside, and applies 30% residual packet loss throughout the yard. This is a prescribed reliability stress case, not a model of a particular emitter. Recovery follows the same queued-message and gossip behavior as normal operation.

## Measured comparison

Each transport was run for 150 simulated seconds under healthy, degraded and disabled communications, with radio seeds 0, 1 and 2. Geometry, initial poses, routes, physics and policy were identical across cases. Seeds vary radio backoff and residual loss; they do not represent three different facilities. The disabled repeats are identical because radio randomness cannot influence motion.

| Transport / case | Runs completing 8/8 | Mean physical completion | Mean duplicate inspections by 150 s | Mean packet delivery | Completion knowledge at 150 s |
| --- | --- | --- | --- | --- | --- |
| Direct / healthy | 3/3 | 48.3 s | 0.0 | 96.3% | 100% |
| Direct / degraded | 3/3 | 43.4 s | 1.3 | 79.2% | 100% |
| Direct / disabled | 3/3 | 38.4 s | 9.0 | No transmissions | 35.4% |
| Reticulum / healthy | 3/3 | 47.1 s | 0.0 | 86.4% | 100% |
| Reticulum / degraded | 3/3 | 46.3 s | 1.0 | 79.8% | 100% |
| Reticulum / disabled | 3/3 | 38.4 s | 9.0 | No transmissions | 35.4% |

Physical completion means every marker was inspected by at least one rover. Duplicate inspections are additional inspections by other inspectors, counted over the entire 150-second run. Completion knowledge is the fraction of all 48 rover/marker completion records known locally. Packet delivery is successful receiver deliveries divided by receiver opportunities (five per broadcast), excluding frames that never left a queue. Queue overflow is separately reported.

Communications eliminate or reduce duplicate work and distribute the full completion record in these runs. They do **not** improve physical completion time with this policy and layout: independent patrols finish sooner. Extra coordination, claims and changed routes can delay coverage. Healthy and degraded results are not forced into a preferred ordering. These small, uncalibrated experiments are evidence of the simulation's behavior, not a general networking performance claim.

Full per-seed metrics, including collision pairs, bytes, airtime, route changes and completion times: [direct comparison](DIRECT_COMPARISON.json), [Reticulum comparison](RETICULUM_COMPARISON.json). [Causal evidence](CAUSAL_EVIDENCE.json) pairs delivered packet records with route decisions and confirms the decision preceded the recipient's first local sighting. All remote route-change claims in all 18 runs were checked against their receiver's actual delivery log.

The dashboard's global rover poses and physical inspection count belong to the observer. The marker map displays R1's local evidence only; those observer fields are never passed to the agents. Packet trails persist for 0.6 seconds for readability and depict delivery/loss outcomes, not beams or real RF rays. The highlighted degraded region is only the prescribed noise zone.

## Structure and verification

- `scene.py`: reproducible primitive vehicle/yard MJCF, nine cameras.
- `agent.py`: local beliefs, claims, patrols, inspection and motion policy.
- `radio.py`: bytes, airtime, queues, propagation assumptions, collision/loss and delivery.
- `reticulum.py`: isolated real-stack processes and custom interface bridge.
- `simulation.py`: local sensor boundary, physical actuation, evaluation and replay logs.
- `visuals.py` / `cli.py`: native viewer, previews and synchronized video.
- `tests/test_rovers.py`: packet/airtime, half-duplex/capture/budget, evidence isolation, physical sensor occlusion, steering clearance/contact-driven turns, three full missions and optional real-stack round trip.

Validation: **16 tests passed**, including the existing hexapod tests; **18/18** separate 150-second rover comparison runs completed; both native direct and Reticulum viewers passed macOS smoke tests. The implementation preserves the original hexapod entry point and assets. Generated trajectories and MJCF stay in ignored `outputs/` and `build/`; checked-in images and MP4 use Git LFS.

## References

- [MuJoCo overview](https://mujoco.readthedocs.io/en/stable/overview.html): physical simulation scope.
- [Semtech SX1276/77/78/79 datasheet, §4.1.1.7](https://cdn-shop.adafruit.com/product-files/3179/sx1276_77_78_79.pdf): packet airtime formula.
- [Reticulum interface manual](https://reticulum.network/manual/interfaces.html#rnode-lora-interface): RNode/LoRa context; this demo uses a simulated custom interface.
- [Semtech LoRa calculator](https://blog.semtech.com/lora-calculator-now-available-on-lora-developer-portal): radio parameter and airtime context.
