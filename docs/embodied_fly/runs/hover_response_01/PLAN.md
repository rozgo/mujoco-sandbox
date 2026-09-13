# Phase-matched frozen feedback probe

User-approved September 13 after distinguishing 500 Hz control from wingbeat frequency.
Use frozen pilot05 and the accepted instantaneous 1 kHz plant. Warm up to six
seconds, select four measured wing phases, then initialize five matched branches
per phase: control, height +/-0.5 mm, vertical velocity +/-10 mm/s. Identical
neural memory, targets and remaining physical state within each group. All five
branches use the same reset procedure. Run .3 s without further reset or teaching.

Measure the first sustained wing-command difference, wing-generated scalar lift
difference (before passive body drag), and signed mean lift corrections over
20/50/100/200 ms. Thresholds are three consecutive 2 ms samples above 1e-5 action
RMS or .005 bodyweights. Lift is sampled on the last of two physics ticks; it is
not the integrated whole-action force. Phase selection is observer-only and not
an actor input. Preserve full traces, hashes, source commit and all phases.
This is local feedback evidence, not general bandwidth identification or a claim
that every response is corrective. No optimizer updates.
