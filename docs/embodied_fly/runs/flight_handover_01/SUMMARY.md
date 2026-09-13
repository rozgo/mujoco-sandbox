# Neural history alone does not rescue wing control

Source `d78d007`, readout01 checkpoint, seed 57001. Four pairs start with exactly
the same **50 ms of physical expert actuation** while the actor observes the
causal sensor stream. At handover, one run retains neural memory; the other
clears only that memory. Both then execute **100% student commands for 100 ms**.
No physical state changes at handover. Native MuJoCo runs at 20 kHz with 5 kHz
control; the full graph runs on CUDA.

All physical states and observations through the handover frame match bit for
bit in each pair. Captured per-frame fractions and actions prove that the teacher
controls the first 250 actions and the student alone controls the remaining 500.
The teacher continues producing counterfactual labels for recording; these are
not used in the executed action after handover. Captured neural state is nonzero
in the retained condition and exactly zero in the reset condition. Model/state
hashes, finite states, bounded actions and causal previous-action feedback pass.

| Speed | Retained memory: first envelope failure after handover | Cleared memory: first failure | Retained / cleared root RMSE after handover |
| --- | ---: | ---: | ---: |
| Hover | 4.8 ms | 6.8 ms | 6.136 / 6.331 mm |
| 5 cm/s | 12.4 ms | 12.2 ms | 7.720 / 5.114 mm |
| 10 cm/s | 17.6 ms | 12.8 ms | 4.101 / 8.846 mm |
| 20 cm/s | 17.6 ms | 11.4 ms | 9.458 / 7.538 mm |

Every run leaves the original height/orientation/tracking envelope; zero MuJoCo
warnings occur. RMSE covers the full 100 ms after handover, including failure.
Retained memory changes behavior and delays failure in some cases, but it does
not establish stable flight or a general advantage. This is a four-pair diagnostic,
not a robustness benchmark or proof of biological brain reconstruction.

Retained/reset setup: **5.450137 / 5.599649 s**. Collection:
**11.957107 / 11.911076 s**. In total, **6,000 physical transitions / 1.2 aggregate
simulated seconds**; no training or video rendering. Both conditions are
explicitly ineligible for teacher-free flight acceptance because of the expert
prefix. The original unassisted acceptance procedure stays unchanged.

Together with the phase-coverage trial, this evidence argues against spending
another pilot solely on startup weighting. Next allow the existing nonlinear
motor decoder to learn from actual motor-cell states, with ground retention and
physical evaluation. No new runtime controller or observation bypass is required.
