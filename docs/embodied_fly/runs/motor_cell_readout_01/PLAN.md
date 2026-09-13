# Motor-cell readout 01 — declared before execution

The preceding constrained output fit still misses the sign of a real near-limit target. Test a richer decoder input while retaining the same measured graph, neural state, canonical body and original actor weights.

Add one zero-initialized linear map from the existing normalized 815 motor-cell activities to six wing-logit corrections: 4,896 parameters. The existing final tanh bounds the total commands. This is a feedforward part of the same motor decoder, with no additional memory, utility selector, observation bypass, force helper or separately selected policy. Zero initialization preserves the parent function. Every original weight, including all original wing/body output weights, stays frozen. Only the new readout is fitted. One checkpoint still controls all 78 actuators for all three commands.

Use exactly the five source histories, pre-fall masks, 8:8:1:1:1 history weights, fourfold startup weight, ridge grid and selection rule declared for state_readout02. The only intended learning change is the feature/readout location: 815 normalized motor cells rather than the compressed 256-dimensional hidden representation. Cache each trajectory member once, with labels verified bitwise identical to the previous loader.

Unassisted evaluation: five seconds per stand/walk/hover, seed 72007, unchanged strict task and posture gates, full neural capture and video. No success claim from label fitting alone. Parent retention02 remains preserved. A successful pilot would still need wider physical validation and ground-posture improvement before release.

```sh
MUJOCO_GL=egl uv run --project experiments/embodied_fly --locked python -m embodied_fly.motor_state_fit \
 --resume assets/embodied_fly/diagnostics/state_hover_retention_02.pt \
 --graph ../fly-survival/outputs/fly_survival/malecns \
 --ground outputs/embodied_fly/state_hover_retention_02_evaluation \
 --hover outputs/embodied_fly/state_hover_reference_02 \
 --correction-capture outputs/embodied_fly/state_readout_01_evaluation \
 --correction-capture outputs/embodied_fly/sweep_probe_01_evaluation \
 --startup-weight 4 --feature-layer motor \
 --output outputs/embodied_fly/motor_cell_readout_01
```
