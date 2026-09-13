# State-based hover reference: physical development probe

Training-only commands derive oscillation phase from measured wing angle and
speed. No elapsed timer enters the reference; no physical parameter changes.
Five gain profiles each run three airborne worlds for five seconds, starting
with still wings. This is CPU reference testing, **zero learning updates**.

| Profile | Root error, mm | Late vertical span, mm |
| --- | ---: | ---: |
| Initial | 4.397 | 8.823 |
| Gentle | 3.176 | 0.320 |
| Gentle, faster energy correction | 1.742 | 0.315 |
| Soft | 5.337 | 0.321 |
| Selected medium, faster energy correction | 1.209 | 0.314 |

Values shown are the first world; the other two starts produce almost identical
errors due to model symmetries. This is parameter selection, not independent
generalization evidence. Root error includes all five seconds; vertical span
uses seconds 2–5. The soft profile fails the 5 mm hover-error gate. All profiles
remain upright with no prohibited support or numerical warnings.

Selected teacher constants: 12 Hz nominal frequency, energy correction 60/s,
height gain 0.4 rad/cm, vertical-speed gain 0.02. A bounded state-triggered
startup leaves zero wing speed. The subsequent deployed actor receives neither
this algorithm nor its phase; it must learn control through MaleCNS activity.

37,500 physical action transitions, 75 aggregate simulated seconds. Measured
profile stepping totals 45.564123 seconds, plus 1.587636 seconds setup. The
preceding three-task reference used another 7,500 transitions and 10.475439
seconds capture, with 3.255465 seconds setup. That reference's walking teacher
fails its ground gates; its successful hover alone motivated this gain probe.

The run started from uncommitted source at parent d405ed6. Exact initial source
snapshots are retained under the preceding state_hover_reference_01 folder;
the probe script and its five explicit gain overrides are preserved here.
Hashes bind all raw captures and archived source snapshots.
