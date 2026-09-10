# Wind operator checkpoints

`fno.pt` and `pino.pt` were trained for this repository using the official NeuralOperator library. They are PyTorch tensor/state dictionaries loaded with `weights_only=True`; Git LFS stores the binary payloads.

- Four Fourier layers, width 32, `n_modes=(16,16)`.
- 603,041 parameter elements per model, including complex spectral weights.
- 64² supervised vorticity trajectories; PINO additionally uses a 128² PDE residual.
- 64 matched epochs per model, trained on the user's RTX 4090.
- Upstream Hermitian-symmetry fix pinned in `uv.lock`; TF32 disabled.

[Exact hashes, training/validation seed splits and selected epochs](../../docs/wind/manifest.json) · [Training and evaluation instructions](../../docs/wind/README.md) · [Measured results](../../docs/wind/RESULTS.md)

These weights approximate this demo's synthetic 2D wind family. They are not a general fluid model, a robot policy or a vendor flight controller.
