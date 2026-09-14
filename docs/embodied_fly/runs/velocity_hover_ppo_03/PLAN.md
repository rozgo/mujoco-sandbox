# Hover PPO: informative reward for initially large horizontal drift

The user's continued-improvement authorization applies. Run 02 enforces small
accepted PPO steps and modestly improves survival, but increases displacement
and does not reduce matched-window velocity error. It is not promoted as hover.

Keep run 02's bounded optimizer, LR 3e-7, network, 32 worlds/16 physics threads,
physical plant, observation/action schemas, zero commands, all other rewards,
critic recipe and light imitation anchor. Start from the same original parent
to avoid inheriting run 02's increased climb. This is a fresh PPO optimizer and
critic for a changed reward; no hidden moment/value transfer across objectives.

One deliberate reward change: horizontal tracking width **0.5 -> 2.0 cm/s**.
The parent typically drifts at about 3.25 cm/s. At 3.0 cm/s, the original
horizontal reward is 0.027; slowing to 2.5 cm/s increases it by only 0.011.
The new width increases that difference to 0.083, over seven times larger,
while retaining the unique maximum at zero speed and the same [0,1] range.
Vertical width, angular objective, alive/upright rates and one-time failure
penalty remain unchanged. This is a training scale change, not a relaxed
physical success threshold. See reward_scale_audit.json for frozen-parent data.

600 measured training seconds, same deterministic midpoint/final starts 0,1,8,9
and original milestone (all four ten-second survivals plus at least 20% lower
first-two-second velocity RMS than the original parent). Preserve the prior
trials and videos; stop comparing conditional errors from differently truncated
episodes as if they were equal-length tests. Render and open the new comparison.

```sh
uv run --project experiments/embodied_fly --locked python -m embodied_fly.velocity_hover_ppo \
  --checkpoint assets/embodied_fly/diagnostics/velocity_imitation_30m_01.pt \
  --graph "$MALECNS_GRAPH" --dataset outputs/embodied_fly/velocity_teacher_dataset_01 \
  --output outputs/embodied_fly/velocity_hover_ppo_03 \
  --baseline-dir outputs/embodied_fly/velocity_hover_ppo_01 \
  --seconds 600 --lr 3e-7 --bounded-updates --horizontal-reward-scale 2.0
```
