# Motor correction with less assistance

Source `887ff1f`, seed 71002, continuation from motor-focus01. Same physical fly,
32-world distribution, fixed measured graph, 397-input/78-output actor and motor
objective. Reference assistance drops from80% to25%, with fresh Adam1e-5 and
32-step recurrent chunks. No utility learning, physics change or action mask.
This remains online imitation on visited states; evaluation has no teacher.

RTX 4090 training took **180.300127 s**, setup7.823445 s. Collection and inference
used141.741318 s; backpropagation and optimization38.552273 s. **155 updates /
158,720 physical transitions /317.440 aggregate simulated seconds**. Peak CUDA
allocation9,411,467,264 bytes. Intrinsic cell parameters received finite gradients
and changed; utility/context weights remained bitwise frozen.

Of618 completed assisted episodes,99 reached two seconds and519 left the
training envelope. Per task: stand37/60 timeouts, walk42/59, hover20/499.
Timeout counts across successive quarters of completed episodes were0/2/22/75;
these are assisted training counts with changing task composition, not a matched
autonomous success rate. Every failure retains a short causal physical trace.

All three unassisted two-second cases still failed. Stand/walk/hover first leave
the envelope at0.388/0.478/0.064 s. Walking survives longer than the first pilot
(0.268 s), while standing does not improve. Hover still cannot support its weight.
All failures remain visible in the [complete six-second review](../../../../previews/embodied_fly/motor_focus_02_all_tasks_v1.mp4).
The same checkpoint controls every segment. Actual simulated neural activity and
observer eyes are synchronized at1×; the video was fully decoded, visually
inspected and opened on the Mac.

The two new pilots total **360.924648 s (6m00.925s)** of training, excluding setup,
evaluation, rendering and transfer. They collected324,608 physical transitions /
649.216 aggregate sim seconds in32 worlds. Neither is promoted. These measurements
establish a reproducible motor-only curriculum, not successful fly locomotion.
