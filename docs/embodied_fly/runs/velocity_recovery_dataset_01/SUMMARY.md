# Full reordered recovery exercises

Ten new continuous 71.2-second PID episodes pass all 500 stage checks, without
resets after initialization. Their ten distinct seeded orders retain all 50
stages. Movement/braking pairs stay together; inverse vertical movements climb
before descending to avoid a command-driven floor collision. Eight episodes
start from saved student error states; two from cold starts.

The new episodes are combined with the original ten unchanged recordings.
There are 16 training episodes (0–7 and 10–17), four held-out episodes
(8, 9, 18, 19), and 1,000 passing stage checks across the combined dataset.
Samples use each episode's actual semantic stage timing rather than assuming
the old ordering. The expected training mixture is 50% original data, 12.5%
new cold starts and 37.5% new recovery starts. No held-out trajectory supplies
gradients. These are development validation cases, not an untouched test set.

New physical collection: ten CPU MuJoCo/mjbatch worlds, ten threads, 356,000
actions and 712,000 physics steps. Capture 231.649219 seconds, setup 4.030683
seconds, total including copying, compression and hashing 281.709578 seconds.
The combined replay contains 712,000 transitions; half were reused, not newly
simulated. Physics remains 1,000 Hz, actions 500 Hz. No student was updated
during collection. Smooth random commands are not part of this dataset.
