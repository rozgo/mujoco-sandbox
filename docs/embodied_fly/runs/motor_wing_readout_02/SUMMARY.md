# Corrective wing calibration — retained failure

Collected eight 0.3-second flight sequences with **25% student / 75% inherited
teacher** executed controls. All eight met the declared demonstration envelope.
The 12,000 physical frames took **46.319653 s** to collect after **6.392281 s**
setup. Actual previous-action feedback and every state hash were verified. These
assisted flights are training data, not student successes.

Replayed original and corrective histories through the frozen readout01 actor:
**12.025874 s** and **12.198033 s** total, respectively. Whole episodes 4 and 6
remain excluded from optimization in each corpus. The 10-second fit changed
only the same **1,542 existing wing output parameters**, with no deployed module,
oscillator, or sensor bypass. All other actor parameters remain identical.

RTX 4090 optimization took **10.000270 s**, setup **1.433782 s**, evaluation/save
**0.035870 s**. It made **19,163** minibatch updates using **18,000 distinct
training frames** across equally sampled corpora. The **19,622,912** sampled
examples are reused optimization samples, not fresh simulated experience. There
were no live physics worlds during the fit. Original-history validation MSE
changed **0.006163→0.006403**; corrective-history MSE **0.014430→0.008380**.

Both unassisted flight cases still fell: hover/forward RMSE **19.854/12.783 mm**.
First-30-ms sampled upward passive force was **0.0824 body weight**, below
readout01's 0.2306; offline improvement did not transfer into stable flight.
Five of six fixed walking cases stayed stable; slow walking fell. Continuous
walk/stop/resume stayed stable with permitted support, but stopping failed its
tracking gate. Since ground evaluation masks wing outputs and no other weights
changed, differing ground trajectories cannot establish learned ground change.

Do not promote this checkpoint. Preserve angle01's reviewed ground video and
the online01 walking/braking reference. The next bounded diagnostic examines
compliant wing-stop overshoot with unchanged actuation and aerodynamics; it is
a separate physical-model variant, not an established remedy.
