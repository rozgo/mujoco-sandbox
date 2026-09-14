# Next: align the objective with coordinated hover before more PPO

Status: proposed after run13; **no new reward or further training has run**.

The planned pause for diagnostics is in effect. Exact-action physical replay
already identifies a concrete objective mismatch: the original reward prefers
the run13 final flight (41.4673) over the retained parent (33.8093), although
total velocity RMS is21.7% worse. Vertical reward increases9.9248 while horizontal
reward decreases only1.8552 over ten seconds. The scorer rewards the tradeoff
that the combined-hover evaluation rejects. This is stronger evidence for
auditing the objective first than for immediately changing the neural interface.
It does not establish that the reward is the only remaining limitation.

1. Keep both policies: run11 midpoint remains preferred overall. Run13 final
   is the proposed experimental parent because startup, later climb and
   withheld vertical recovery improved substantially. Preserve its optimizer,
   critic, same full brain, physical model and action/observation interface.
2. Test a single coordinated velocity term on the saved flights, including
   run11, run12 and run13 and explicit zero/drifting/falling counterexamples.
   A candidate is the existing inverse-quadratic total3D-speed reward, retaining
   the current5mm/s vertical scale for the whole vector. Keep its maximum rate3,
   alive/orientation rewards,100ms causal velocity window and failure penalty.
   Score phase-averaged body motion; do not add wing-angle/phase imitation.
3. Verify that the candidate gives the highest score to coordinated stationary
   hover, does not reward the observed sideways regression as an overall gain,
   and still prefers recoverable flight over early failure. Inspect term totals
   and physical objectives separately. Do not assume a familiar formula passes.
4. Only if that offline objective audit passes, run one~10minute/108-rollout PPO
   trial from run13 final, preserving32 worlds/16 normal+16 recovery starts,
   .0015 exploration,.005 KL cap and light imitation. The reward change and
   corresponding critic adaptation must be explicit. Retain its new vertical
   ability while asking it to brake sideways motion.
5. Compare cold and withheld recovery tests halfway and at the end. Require
   lower total/horizontal motion, four complete normal flights and six complete
   recovery flights, while preserving the final candidate's vertical/startup
   improvement. The combined hover targets remain total RMS<12.76mm/s,
   horizontal RMS<10mm/s and climb<50mm; do not promote on return alone.
6. If a correctly ordered objective still yields no physical gain, keep training
   paused and measure phase-conditioned sensing/control authority: can the
   current motor features distinguish opposite velocity errors at matched wing
   phases, and can the existing wing actions generate the needed braking?
   Change the encoder/readout only if that diagnostic identifies a limitation.

This is not an unannounced repeat of run08. That earlier vector reward used
20mm/s on all axes, making vertical precision much looser. The proposed audit
keeps5mm/s and tests the saved behavior before spending another training block.
