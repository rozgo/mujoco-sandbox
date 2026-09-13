# Hover teacher clock audit

With physical state and all397 instantaneous sensor inputs held unchanged,
changing only the existing teacher's timer changes its normalized sweep commands
from+0.5227 to-0.6278 (range1.15044). The same effect appears on both wings.
No physics steps, training updates or student actions are performed.

This establishes timer dependence of the current teaching labels. It does not
establish that a recurrent actor cannot learn the reference: recurrent history
can encode phase. It is also not proof that this is the only hover-learning
problem. A state-based teaching reference is worth testing because it would
remove this extra timing requirement without changing the deployed actor/body.

The audit uses the same canonical wing_motion preset on Mac. Its compiled
fingerprint is the established Mac variant; it does not replace the GPU training
body or bypass checkpoint verification. Setup3.320554s; audit0.004622s. The
integrated physical state, controls and previous action remain unchanged.

Reproduce:

```sh
uv run --project experiments/embodied_fly --locked python -m embodied_fly.hover_clock_audit \
  --teacher assets/embodied_fly/teachers/walking.npz \
  --output outputs/embodied_fly/hover_clock_audit_reproduction
```
