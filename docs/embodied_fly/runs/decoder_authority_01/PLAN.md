# Local decoder authority before another learning change

Start **2026-09-14 21:17:28 UTC**. Follow the run-14 control diagnosis plan.
The previous turn produced measured learning and failure evidence; the broader
learning goal remains active.

- Preserve run 13 final and its exact full body/brain recovery bank 04.
- Probe six existing wing-readout output biases and a common multiplicative
  change to the existing left/right sweep output weight rows. The latter scales
  the residual readout rows, not the base decoder or an external force.
- In 32 worlds, compare two histories at a time, each with an unchanged control,
  seven positive/negative parameter pairs and a duplicate unchanged control.
  Perturbation magnitude 0.001; short 128 ms physical continuations.
- Histories: cold 0/0.2/0.6 s and warm 2/2.008/2.016/4/6 s. All use training
  episode zero; this local diagnostic is not a held-out capability result.
- Every action still runs the full MaleCNS actor. A final-layer diagnostic hook
  is tested against actual copied-layer parameter changes. Nothing is optimized
  or added to the deployed controller. Verify original parameters unchanged.
- Record every 1 kHz wing-force tick, final velocity changes, failure margins,
  duplicate errors and finite-difference authority. Do not infer controllability
  from matrix rank alone; compare signal size, coupling and operating condition.
- Use 16 CPU MuJoCo/mjbatch threads and RTX 4090 neural inference. Report setup,
  parent-history collection and probes separately from learning.

In parallel, check primary literature and code for interventions relevant to
startup retention and effective exploration. Select a bounded learning change
only after inspecting the physical probe, rather than replacing the stack by
algorithm name. The user's request to consult research was added during this
diagnostic.
