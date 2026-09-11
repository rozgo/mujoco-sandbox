# Working across the Mac and GPU machine

The shared repository is [rozgo/mujoco-sandbox](https://github.com/rozgo/mujoco-sandbox), with SSH remote `git@github.com:rozgo/mujoco-sandbox.git` and default branch `main`. Use commits to move source, configuration, selected checkpoints and finished media between machines. Existing demos and their recorded versions remain in history.

## New checkout

Install Git, Git LFS and uv, and configure the machine's GitHub SSH access. Then:

```sh
git lfs install
git clone git@github.com:rozgo/mujoco-sandbox.git
cd mujoco-sandbox
git lfs install --local
git config pull.ff only
git config push.default simple
git lfs pull
uv sync --locked --extra wind --extra reticulum
```

Omit extras for the base demos. Each machine builds its own `.venv`; never copy a Python environment across operating systems. Use an isolated uv project/lockfile if a future GPU training framework needs incompatible dependencies.

## Before working or training

```sh
git status --short
git pull --ff-only
git lfs pull
uv sync --locked --extra wind --extra reticulum
git rev-parse HEAD
```

Start from a clean checkout. Preserve or commit local work before pulling; do not reset it away. Log the source commit, configuration, random seed, device and dependency versions with each training run. Keep a long-running job's checkout fixed until it finishes; use another worktree for concurrent experiments.

## Hand off completed work

Save selected model weights under `assets/<demo>/` and finished images/videos under `previews/<demo>/`. Save concise experiment reports in `docs/<demo>/`. Stage the specific files belonging to the result, review the diff, and commit them together. `git push` uploads referenced LFS objects through the installed hook.

```sh
git diff --cached --check
git diff --cached --stat
git commit -m "Describe the completed experiment"
git push
```

On the other machine, use `git pull --ff-only` and `git lfs pull`, then verify the commit and checkpoint hashes before evaluation. For work happening on both machines at once, use separate branches with `git switch -c <branch>` and `git push -u origin <branch>`; merge deliberately after review. Do not force-push shared `main`.

Adaptive dog v1 is now the official baseline on `main`, pinned by the
`adaptive-dog-v1` tag. Its checkpoint and video identities are recorded in
[the release manifest](locomotion/OFFICIAL_V1.json). Start subsequent learning
work on a new branch and use new artifact names; preserve the v1 checkpoint,
video, GIF and recorded limitations. The original `experiment/adaptive-dog`
branch remains available as development history.

## Large files and generated data

- `.gitattributes` routes supported mesh, image, video and model formats through Git LFS, including `.mp4`, `.pt`, `.npz`, `.onnx` and `.safetensors`.
- Use `git lfs ls-files`, `git check-attr filter -- <file>` and `git lfs fsck` to check tracking and integrity. A missing LFS download is not a valid model checkpoint.
- Generated datasets, weather caches, intermediate trajectories, raw logs and scratch renders stay under ignored `outputs/` or `build/`. They do not synchronize through ordinary commits; regenerate them from the committed configuration when needed.
- Keep SSH keys, tokens and machine-specific connection configuration outside this repository. GitHub authentication is configured independently on each machine.

## Smoke checks after syncing

```sh
uv run --locked --extra wind python -c 'import mujoco, torch; print("MuJoCo", mujoco.__version__); print("CUDA available:", torch.cuda.is_available())'
uv run --locked --extra wind python scripts/check_wind_backend.py
```

Use the relevant tests and viewer checks from [AGENTS.md](../AGENTS.md) for the change. For headless Linux rendering, set `MUJOCO_GL=egl` for that process. Native macOS viewing uses the existing `mjpython` launchers. Git synchronization does not guarantee identical numerical results across devices; verify policy outputs and physical behavior on the target backend.
