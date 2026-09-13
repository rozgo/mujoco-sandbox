# First physical-reward wing-motion pilot

Source `ec769f9`, seed 65001, brain01 parent. One full MaleCNS actor; 32 native CPU/mjbatch physics worlds and 16 threads, RTX 4090 graph inference and PPO. Physics 5 kHz, control 500 Hz. No teacher controls live rollouts. Ground rehearsal is explicit and separately timed.

Training took **182.873925 seconds** after 5.321930 seconds of setup. Collection 108.606550 seconds; PPO optimization 58.779321 seconds; ground rehearsal 15.401566 seconds. Collected 129,024 physical transitions / 258.048 aggregate simulated seconds, with 211 PPO updates. Peak CUDA allocation 9,572,085,760 bytes.

Of 716 completed episodes, 127 reached the 0.5-second timeout without leaving the training envelope, and 589 failed. Timeouts are not tracking success. Four consecutive quarters had 37 / 31 / 21 / 38 survivors out of 179, showing no sustained trend. Both independent two-second mean-action flights failed; hover falls below 8 mm at 0.244 seconds, earlier than its imitation parent.

The actual learned stochastic policy was also tested at predetermined seeds 66001–66004 using source `d2f3149`. All four two-second hover trials failed. Every saved training failure trace was hash-checked and finite on the GPU host; local evaluation captures and checkpoint hashes were verified after transfer. No promotion or blind extension of this pilot.
