# Continuous velocity PID teacher: development run 1

Reference controller only; no actor or critic training. One continuous
71.2-second physical episode with all movements, no resets.

29/50 declared stages pass. The complete exercise is not accepted
under the declared gates. All raw trajectories remain preserved.

Capture 65.180640 s; total wall 75.123324 s.
Physical plant: wing_motion_heading_v3, independent yaw driven by measured
wing pitch, unchanged body mechanics. Later teacher tuning changes feedback
gains while preserving this model. The failed stages and exact gains are in report.json.
