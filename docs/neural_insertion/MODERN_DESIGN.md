# Insertion mechanism, modern interpretation (design v2)

October 6, 2026. User direction: build a current machine, not a 2019 one. This
is our educated guess at a present-day thread-insertion head, informed by what
has been published since the 2019 Neuralink paper. It replaces the first-pass
mechanism ([E2E.md](E2E.md)) for the end-to-end cycle; that pass and its
findings stay on record.

## What is documented (and where)

- Thin laser-milled tungsten-rhenium needle, about 40 µm tapering to about
  24 µm, with a ledge at the tip that catches a loop at the thread's end; the
  latest needle reportedly passes through the dura without its removal.
  (Press coverage of Neuralink updates; 2019 paper.)
- A swappable needle-and-pincher cartridge; threads peeled one at a time from a
  backing beside the insertion site; a pincher holds the loop's neck against a
  cannula during transport; very fast retraction (up to 30,000 mm/s²) snaps the
  needle free of the embedded thread. (US 11,103,695, "Device implantation using
  a cartridge".)
- About 1.5 s per thread in the current robot. (PRIME study update and press.)
- Surface tracking and tissue-relative control for pulsating cortex: optical
  coherence tomography (Neuralink patent US 11,712,306) and, in academic work, a
  harmonic observer with constrained MPC reaching 1–12 µm relative placement in
  simulation (arXiv 2608.08860).

## Our design (inferences marked)

1. **Anchored thread, short free span.** The thread's root is fixed to an
   implant carrier on the tissue a few millimetres from the target; it lies on a
   backing film with its loop presented over the film's edge. Only about 8 mm is
   free. (Inference from the peel-from-backing cartridge and implant geometry.)
2. **Needle with a ledge.** A 50 µm tip section passes through the loop; a
   ledge just above it presses on the loop's far edge, so advancing pushes the
   loop into tissue and retracting slides out of it. (Documented concept;
   dimensions scaled to our thread, below.)
3. **Rotary pincher.** A light arm on the needle carriage swings in and pins the
   loop's neck against the needle with a limited force for the move to the
   target, and swings away before insertion. (Documented concept; our geometry.)
4. **Fast insert, snap retract.** The needle drives in quickly and retracts at
   high acceleration; the tissue holds the loop. (Documented.)
5. **Puncture model, not a pre-formed tube.** A needle-tissue force model:
   dimpling stiffness to puncture, then cutting and shaft friction, with tissue
   grip on the inserted thread; parameters illustrative until measured.

## Compromises we state openly

- Real threads are polymer ribbons a few micrometres thick. Ours is a 40 µm
  round discrete elastic rod: thinner segments would force a still smaller
  timestep. Needle and loop are sized to our thread (needle tip Ø 50 µm, shank
  Ø 80 µm, loop radius 100 µm), so they are larger than the documented ones.
- The backing film is a surface the thread rests on; adhesion and peeling
  forces are not modelled yet.
- The dura is not yet a separate layer in the puncture model.

## First complete cycle (October 6)

`python -m sixlegs.neural_insertion.modern` runs pick, pinch, peel, transport,
align, pincher release, insertion and snap retraction with scripted yardstick
motions, no disturbances, and the provisional puncture model. All checks pass:

| Measure | Result |
| --- | ---: |
| Loop lifted with the needle in the peel | 1.46 mm |
| Needle tip over the target before insertion | 7.5 µm |
| Puncture | yes; peak needle force 2.0 mN (the model's threshold) |
| Loop depth at full insertion | 1.67 mm |
| Loop depth after the snap retraction | 1.65 mm |
| Loop to withdrawn needle tip | 2.65 mm |
| Loop centre from the target | 42 µm (mostly the loop's offset from the needle axis) |
| Robot contacts with the tissue | 0 |

0.9 s simulated in 38 s on one Mac core (about 40 times slower than real
time, against 600 for the first-pass scene): the anchored 6-segment thread has
15 degrees of freedom instead of 99.

What it took, each found by running the cycle:

- **Pinch geometry.** A jaw arcing in along the thread's direction rode over
  the ring or pinned thread on the film. The jaw swings in from the side at the
  ring's height, slightly below the wire's centre, and pins the ring's near wire
  against the needle with a torque-limited 100 µN.
- **Contact pairs.** MuJoCo skips contacts between a body and its parent, so the
  jaw (on the needle carriage) passed through the needle until explicit
  jaw-needle pairs were added.
- **Slack.** A thread lying straight from its anchor goes taut the moment its far
  end is lifted (about 9 mN pulled the loop off). As in a real cartridge, the
  implant sits near the target, the thread lies on the backing running away
  from it, and the robot picks the far end and brings it back.
- **Puncture track.** Only the loop and two segments could pass the tissue
  surface; the third was pulled into the track and blocked, and the tension blew
  up the simulation. Four loop-side segments now follow the needle through.

Still to do: disturbances, a calibrated puncture model with a dura layer,
adhesion and peeling from the backing, several sites, and learning each phase.
