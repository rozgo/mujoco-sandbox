# One general decoder for the whole fly

User direction: remove the isolated wing decoder; legs, antennae, mouth and wings
must use the same decoder. Show the current learned behavior and prediction
result in a new video. Effort started September 14, 23:52:33 UTC.

The active migration combines the previous 256-unit motor layer and 128-unit
correction layer into **one 384-unit hidden layer and one dense 78-output layer**.
There is no wing selector, separate branch or output addition in live decoding.
Every output can learn from every hidden unit. The initial weight values preserve
the preceding function; zero initial connections are trainable, not masks.

The decoder consumes the same 815 MaleCNS motor-cell values. Generic preprocessing
retains two views of those values: LayerNorm and standardized/clipped LayerNorm,
yielding 1,630 input features. Keeping both preserves the prior input clipping
exactly, including outlying states. Neither view corresponds to a body part;
both enter the shared dense layer. No extra recurrence or direct sensor bypass.

This is an algebraic consolidation, **zero optimizer updates**, not new learning.
The historical checkpoint remains preserved. The active checkpoint records a
fresh-optimizer requirement because the decoder topology changed. Full-body
training can update all decoder parameters, including normalization and every
output row; cached training must use raw motor-neuron values, not stale normalized
features. The old wing-only PPO CLI remains a historical experiment, not the
recommended training path for this checkpoint. Future model-guided work must use
the full-body decoder and assess its complete motor outputs.

`fresh_velocity_actor` now initializes the shared decoder by default. Its explicit
`--legacy-wing-readout` flag reproduces old initializations only. Old policies
continue to load with their recorded architecture; they are not silently changed.

Verification: exact real-number function preservation, numerical action checks
including clipped inputs, unchanged upstream tensors, no wing-branch tensors in
the new checkpoint, gradients on all 78 rows and newly available connections,
then four full ten-second MuJoCo flights against the retained policy's captures.
The video must identify persistent drift and distinguish learned flight from
world-model prediction. Neural images show actual simulated latent activity
binned at measured cell locations; they do not claim physiological spikes.

Results, measured times and the video link will be recorded after physical review.
