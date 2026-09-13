# State readout 01: better reference fit, failed autonomous hover

Source `354bf2e`, parent `state_hover_retention_02`. Only six existing wing-output rows and biases changed (1,542eligible parameters). All upstream state, other72output rows, graph routing and canonical body remain unchanged. The checkpoint is one ordinary motor-only actor for stand/walk/hover.

The fit uses real frozen-actor features from three existing5s histories: parent standing/walking and the measured-state hover reference. First4s per history fit the readout (6,000frames); last1s per history select regularization (1,500frames). These are temporally related windows, not independent episodes. No new physical transitions were collected. Labels are parent ground outputs and reference wing commands; no teacher controls evaluation.

Setup4.521751s, frozen GPU replay13.565234s, ridge fitting/packing/save0.928369s; total19.015356s. Five regularization values are retained. Selected0.01; output weight/bias change L2norm0.06744. Last-window wing-action MSE is0.000000550stand,0.00001117walk,0.0002194hover. This is an offline fit result, not proof of learned flight.

The full unassisted five-second three-command test (seed72003) takes7.295372s setup +32.408363s capture. Standing and walking remain upright with permitted support, but resting-wing peaks are0.2703/0.4049rad, exceeding the0.2rad gate. Hover falls, root tracking RMS19.220mm. All strict task gates fail, with zero numerical warnings. The candidate is archived and **not promoted**; retention02remains the development parent.

The [current-state boundary audit](boundary_feedback_audit.json) exposes a concrete error: at0.20s, sweep angles are about-1.45rad and the actor commands-0.32/-0.31, while the existing measured-state hover reference requests+0.47/+0.48to reverse. At0.50s the wings remain against the lower limit with outward commands. On the actual0.002s state, startup commands0.22/0.27also fall below the reference0.87/0.91. These are recorded-state diagnostics, zero new integration or learning; they do not establish a unique root cause.

The next data change should add the actor's real startup and near-limit mistakes with current-state corrective targets, alongside the successful reference and retained ground histories. Current fitting only covers the successful reference's narrow hover states. This proposal is not yet implemented or run.

[Complete video](../../../../previews/embodied_fly/state_readout_01_all_tasks_v1.mp4):15s,750frames,1600×900,50fps,1×. Fully decoded, visually inspected and opened; rendering/encoding60.334684s. An independent verifier reconstructs the selected ridge solve and validation metrics, confirms frozen state and physical contracts, and checks every saved bounded action and causal feedback step.
