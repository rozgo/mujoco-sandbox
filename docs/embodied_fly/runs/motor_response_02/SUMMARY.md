# Wing response after paired supervision

Frozen checkpoint 08 is probed on its own recorded histories using the same
procedure as response01. Setup takes 4.272073 s; diagnostic takes 4.017320 s.
75 parallel neural sequences, zero physical transitions and training updates.
Original-history action replay agrees within 4.62e-7.

Across 48 ground angle checks after 25 held-input updates, mean restoring gain
is -0.03508 action/rad; the target is -0.66667. Six checks have the wrong sign.
Mean velocity gain is -0.001147 action/(rad/s), versus target -0.005; six of
48 checks have the wrong sign. Response01 on checkpoint06 had means -0.02245
and -0.0002404, with seven and sixteen wrong-sign checks respectively.

These summaries use each checkpoint's own physical histories, so this is a
descriptive comparison, not an isolated causal estimate of the training change.
Neither probe advances physics. The physical evaluation remains decisive:
08 holds wing angles less accurately and does not replace preferred06.
Input scaling and body physics were unchanged.
