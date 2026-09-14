# Next: local wing-decoder authority across startup and established flight

The continuing learning goal remains active. Do not promote run 14 or run a
longer unchanged PPO continuation. Run 11 is preferred overall; run 13 final
is the stable vertical-control reference for the next diagnosis.

Evidence: run 14 final completes six warm recoveries but fails all cold starts
at 0.796 s. Exact replay measures only 94.2% of body weight in raw lift just
before failure, versus 100.8% during established recovery. Wing motion persists.
The failure needs state-dependent regulation, not a global force increase.

1. Preserve graph, physical model, command/action interface and all accepted
   policies. Use cold initialization and the saved parent-generated body/brain
   histories. Restore complete states; introduce no live pose corrections.
2. Test small positive/negative changes to existing wing-readout parameters in
   diagnostic copies: six output biases and collective sweep output gain.
   These are parameter sensitivity probes, not a new deployed controller or
   force offset. Keep every other parameter fixed and report perturbation sizes.
3. Compare short physical continuations at several wing phases. Measure actual
   per-tick lift, lateral force, velocity change, uprightness and failure margin.
   Use duplicate unchanged controls to bound numerical variation. Determine
   whether local parameter changes provide independent support and braking,
   and whether their sign changes between cold and warm histories.
4. If physical authority exists, probe signed velocity errors through the same
   full MaleCNS path at matched phases and inspect the current readout response.
   The earlier run-08 probe already established input transmission; do not
   mistake its rank result for proof of adequate current closed-loop control.
5. Choose the learning intervention from the result: improve phase-dependent
   decoder learning if the required actions are reachable; change sensory
   representation only if the probe identifies missing distinctions. Retain
   cold-start support explicitly while improving warmed-up velocity regulation.
   Any reward, exploration or trainable-parameter change must be recorded.
6. Before another extended block, require a bounded candidate to complete the
   four cold flights and six recoveries, with reduced combined motion and no
   startup regression. Keep unsuccessful probes and their measured cost.

The immediate task is this bounded authority diagnosis. No additional learning
change is silently included in the completed run-14 results.
