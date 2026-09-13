# Motor focus 10: higher-rate ground imitation

Not selected. Preferred motor_focus_06 remains preserved; it is still not a
completed motor release. This 180.398 s continuation of09 uses fresh Adam1e-4,
32 worlds (16 stand /16 walk), the same canonical wing_motion body, full
motor-only actor, reset disturbances and corrective imitation losses.

Collected132,096 physical transitions (264.192 aggregate simulation seconds),
129 updates,128 completed episodes with5 physical failures. Setup9.531s;
collection/forward130.144s;backward/optimization50.245s. Peak CUDA allocation
13,824,745,472bytes. Physics remains CPU MuJoCo/mjbatch,5kHz; actor and learning
run on CUDA,500Hz actions. Utility/intention weights are unchanged.

Five-second independent development evaluation (seed72001, no teacher or
resets) keeps both ground cases upright on permitted supports. Standing speed
RMSE improves to0.0734cm/s and yaw to0.4621rad/s, but wing RMS0.2027rad and
maximum0.3161rad fail the initial-form requirement. Standing height loss6.275%.
Walking speed RMSE worsens to1.0101cm/s for a1cm/s command: the fly effectively
stops walking. Walking wing RMS0.1512rad/max0.2747rad also fails. Hover falls.
No complete task passes. Capture32.679s after6.022s setup;zero MuJoCo warnings.

The complete15s film retains allthree cases at1x. All750 encoded frames decoded,
six sampled frames inspected, then opened on the local Mac. Capture/checkpoint/
model/failure hashes and causal previous-action feedback are verified in the
adjacent evidence. The failed candidate is retained rather than promoted.
