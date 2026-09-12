# Physical-outcome PPO 01

A developmental shared graph controller, **not a completed fly survival policy**.
All six deterministic cases kept upright valid-foot support. Stopping still
fails; right-turn motion is less erratic but tracks its target too weakly.

| Measured quantity | Result |
| --- | --- |
| Training wall time | 301.196457 s |
| Setup | 5.697281 s |
| Live collection (physics + inference + bookkeeping) | 142.704298 s |
| PPO optimization | 139.156288 s |
| Explicit imitation rehearsal | 19.193652 s |
| Independent full-body worlds | 32 |
| CPU physics threads | 16 |
| Neural computation/learning | RTX 4090 |
| Actual physical transitions | 278,528 |
| Aggregate simulated experience | 557.056 s (9.28 minutes) |
| Total throughput including learning | ~925 transitions/s |
| Physics / control | 5,000 Hz / 500 Hz |
| Rollout | 64 actions × 32 worlds = 2,048 transitions |
| PPO updates / rehearsal presentations | 1,090 / 34,816 |
| Actor / training-only critic parameters | 2,409,132 / 233,985 |
| Measured graph | 166,700 nodes / 25,582,938 aggregated connections |
| Peak CUDA allocation | 5,342,385,152 bytes |
| Greedy development tests | 6/6 stable, 6/6 valid support, 0/6 original tracking gates |
| Numerical warnings in greedy tests | 0 |

Ancestry: 600.083779 seconds of supervised optimization, then this PPO stage.
This excludes inherited teacher training, teacher-data collection, setup and
separate failed/probe experiments. A single checkpoint drives all six tests;
there is no teacher or gait generator during their evaluation. The terrestrial
curriculum explicitly holds 19 wing/mouth/antenna channels passive while retaining
the full 108-DoF body. These 32 worlds are independent training environments, not
yet interacting agents in the survival arena.

[Full interpretation and comparison](../../LEARNING_JOURNAL.md),
[training report](report.json), [evaluation](evaluation.json),
[matched recorded-state metrics](tracking.json),
[forced-utility diagnostic](forced_rest.json).
