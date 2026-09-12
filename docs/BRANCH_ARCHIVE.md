# Branch consolidation — September 12, 2026

The accepted graphite presentation and complete normal-speed adaptive-dog film
are merged into `main`, together with the paused generative-styling experiment. `main` is the only active local and origin branch.
The three existing release tags remain unchanged.

| Former branch | Preserved tip | Where its history remains |
| --- | --- | --- |
| `experiment/adaptive-dog` | `81fa34c` | Main history; `adaptive-dog-v1` release |
| `experiment/adaptive-dog-gpu-comparison` | `95aa54e` | Main history |
| `feature/adaptive-standing` | `0e344f4` local; `fab7d12` former origin/GPU | Main history; `adaptive-standing-v1` release |
| `feature/moving-supports` | `519afe7` | Main history; `adaptive-moving-v1` release |
| `feature/graphite-presentation` | `b7c26ac` | Main history |
| `feature/cinematic-video-styling` | `c1f4e82` | Main history |

The generative-styling experiment is included in main and remains paused. Its
WaveSpeed SDK project, scripts, docs and selected trial media are preserved for
future work. No new paid generation is submitted as part of consolidation.
See [the setup and review guide](video_styling/README.md).

The GPU's former experiment worktrees remain at their original commits in
**detached HEAD** state, preserving ignored trajectories, render caches and uv
environments. Their branch references have been removed. The primary GPU checkout
uses `main`, matching the Mac and origin. No worktree directory was deleted or
cleaned. Existing replay commands can still point `--trace-root` at those original
capture checkouts. Create a new branch before developing in an archived worktree.

Main's preferred movie is the [124-second normal-speed showcase](../previews/locomotion/graphite/adaptive_dog_complete_v2.mp4).
Prior movies, frozen checkpoint identities and measured limitations are preserved.
