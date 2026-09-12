# Moving-support experiment

User request: “lets merge our progress so far with main, and then lets do this new moving platform experiment”; “lets set our new goal to moving environment effort”.

Accepted standing work is preserved on main at `d652f8f`, tag `adaptive-standing-v1`. This experiment uses `feature/moving-supports`, separate checkpoint/video paths, and the existing isolated uv package with a separate moving-support CLI.

Build a physically actuated deck that translates, yaws, heaves and rocks. The dog remains a free body with the original torque-limited joints. One shared policy should hold its position on the deck, using the same actor/critic architecture and short measured GPU PPO rounds. Mix static standing and walking to preserve learned behavior. Start with a static preview and frozen-policy measurements. No live pose correction or scripted dog motion.

Initial scope: healthy dog on moving supports; all nine bodies retained on static ground. Damaged moving support is an explicit transfer test, not a claimed training distribution. Ideal simulator platform pose/twist provides support-relative preprocessing; the future motion command is never policy input. Existing joint state, body orientation/angular velocity, rays and contact sensors remain. Observer cameras are not policy inputs.

Before tuning: ten-second uninterrupted trials, no evaluation reset. Four trials per case. Require finite state, no simulator warnings, robot torque caps, sampled penetration <=8 mm, gravity tilt <=20 degrees, support-relative drift <=15 cm, mean relative speed <=6 cm/s, >=2 allowed supports for >=95% of settled hold; undesired link reaction <=1 N at every 2 ms substep. Platform motion tracking and actuator saturation are reported separately. Report survival separately from strict passes; retain failed trials. Motion begins smoothly from rest. Static retention uses the existing unchanged acceptance gates, including walking gates.

Separate three comparisons: original frozen observation semantics; the same frozen weights with support-relative preprocessing; newly trained shared weights with that preprocessing. This distinguishes coordinate correction from learning. Match seeds, physical actuator targets and initial states; measured platform motion can differ slightly because the robot physically loads it.

Seeds: training 21/22/23, development 9401, holdout 9407, video 9411. Choose a checkpoint before holdout/video. Start with 60–120 s learning rounds, initially up to ~300 s new training; extend only with measured evidence, recording all attempts and inherited learning time. No hour-scale training.

Output: overview/detail static preview, following/overhead/head views, matched real-time video with measured relative drift and honest failures, reproducible CLI, raw result reports and trajectory provenance, Mac and GPU checks, Git/LFS sync. Work began 2026-09-12 02:37:25 UTC; goal registered 02:37:55 UTC.
