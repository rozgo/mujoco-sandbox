# Additional starts still fail after PPO

Outcome04 uses exactly the same 27-case diagnostic as the parent: seeds98103,
98113,98123, paired noise0/.003/.01, stand/walk/hover, five seconds without
live resets or parameter updates. All 18 ground cases remain stable. All nine
hover cases fail, including the three deterministic ones. Physical initial
states, activation, target heights and seeds match the parent comparison.

Deterministic hover fails at .590, .552 and 1.404 seconds respectively, versus
.760, .570 and 1.968 seconds for the parent. This does not establish a robustness
improvement; the candidate is preserved, not promoted over the parent. The
matched original review remains a useful local PPO result, not general success.

All states and file hashes, bounded controls, causal feedback, initial-state
matches and unchanged parameter hashes are verified. No numerical warnings.
