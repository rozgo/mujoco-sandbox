# Continuous wing feedback: second pilot

The new sensory inputs preserve wing-speed information that the walking-trained
normalization clipped. They do not yet produce physical flight. Both 0.3-second
airborne probes fell; five of six fixed ground cases and the continuous
walk-stop-walk transition retained stable, permitted support. All original raw
tracking gates still failed. The slow ground case fell. Online01 remains the
walking/braking reference.

The actor has 389 inputs and 78 outputs. Six new inputs, scaled by 2,000 rad/s,
enter the existing sensory hidden layer through a zero-initialized 6→128 matrix.
There is no direct motor bypass. The fixed 166,700-cell graph remains intact;
internal cell dynamics and interfaces are trainable. The new matrix adds 768
parameters, for 2,409,900 total trainable parameters.

The full-graph CUDA migration differed by at most 4.18e-7 in action, within the
declared 1e-6 numerical tolerance. Repeated execution of the unchanged parent
also differed by up to 5.37e-7. Both failed earlier bit-equality checks remain
archived. This numerical compatibility check does not relax physical gates.

Eight retained-student episodes collected 12,000 transitions over 24 physical
seconds, including two uninterrupted command-transition trials. All eight were
stable with permitted support. Collection took 64.064438 seconds after 3.046267
seconds of setup. Their executed actions provide ground rehearsal; the flight
targets still come from the inherited expert and its supplied wingbeat pattern.

Training used the RTX 4090 for **60.094962 seconds**, with **32 offline neural
sequences**, 207 updates and 105,984 supervised examples. Setup took 4.369536
seconds; validation 0.595750 seconds. Peak CUDA allocation was 5,176,526,848 bytes.
Walking and flight batches were sampled at their respective 500 Hz and 5 kHz
clocks. Adam was initialized fresh for the new parameter layout. Flight validation
MSE fell 0.463187 → 0.052054; ground retention MSE rose 0.000421 → 0.002731.

At the same initial hover pose, the first 30 ms of the student capture produced
only 0.0526 body-weight mean upward passive force, versus 0.9630 for the expert.
These are 150 replayed control-boundary force samples, not substep averages.
The new six velocity channels remained unclipped. Low wing speed and loss of
altitude remain the physical failure, despite improving supervised error.

Checkpoint: `assets/embodied_fly/diagnostics/motor_flight_probe_02.pt`.
SHA-256: `ab0c4f53ee43579483c026c017d8cd1c9103dbfe12f6da96a8771d2ae4a86503`.
Parent: online01, `9b9ab703483a52d06eb1700172e762b8226e3b4b6597678176a89bd886b6649a`.
Training source: `4af885e`; seed 45001; learning rate 3e-5; 16 burn-in and 16
supervised steps; 25% reset-start batches; unit extra wing MSE on flight batches.

Exact results are in `report.json`, `progress.jsonl`, `hover.json`, `forward.json`,
`walking.json` and `transitions.json`. `wing_diagnostic_prearchive.json` retains the
initial diagnostic with its recorded uncommitted-source flag; the clean-source
repeat is separate. No failed result has been overwritten or promoted.
