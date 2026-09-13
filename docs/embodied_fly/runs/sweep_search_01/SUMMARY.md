# Sweep search 01: no accepted calibration

The declared16-candidate search completed in70.556s plus6.464s setup. It used32physical worlds, eight candidate decoders per batch, two batches, and four matched cases per candidate. Each case ran the full2s without hiding failures:64,000transitions /128aggregate simulated seconds. CPU MuJoCo/mjbatch supplied5kHz physics; the RTX4090 evaluated the brain at500Hz. Peak allocated CUDA memory:811,394,048bytes.

No candidate met the declared combined selection rule. The exported actor retains the parent's exact state; there is no selected behavioral update. The mild bias-only candidate (gain1,bias0.02) retained ground behavior but reduced short hover cost only3.18%, below the declared5% threshold. All32airborne search cases eventually failed.

The best unconstrained candidate (gain1.3,bias0.02) maintained near-starting height longer, but worsened standing-wing posture and drifted forward. Its sweep joints reached their travel limit and stopped producing lift. It remains a rejected diagnostic; see the [full follow-up](../sweep_probe_01/SUMMARY.md).

Independent verification checks all64,000bounded actions and causal feedback, exact initial observations across every candidate, hashes of both complete traces/model/checkpoint, canonical physics, the selection calculation and exact exported weight transform. The traces contain27,030failed world/frames, retained alongside successful frames. No gradient-based learning occurred; this was physical parameter-space search.
