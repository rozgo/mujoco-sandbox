# Fly brain architecture image

This diagram is a snapshot after the faster PID teacher reference passed 50/50 stage checks. It separates that completed reference from the fresh motor student's future training and later utility/multi-fly stages.

The MaleCNS topology and normalized connection weights remain fixed. The implementation includes trainable neuron excitability, leak and bias, as well as the encoder and motor readouts. The figure's dashed teacher arrow represents imitation supervision, not a runtime sensor-to-motor bypass. The core drawing is a schematic, not measured neural activity.

Generator: built-in image generation tool. One initial generation and one targeted correction. No CLI/API fallback. Both prompts are preserved below. The PNG is an explanatory illustration, not a simulation render.

## Initial prompt

Use case: infographic-diagram.
Create one polished, high-resolution landscape architecture infographic, about 2400 by 1600 pixels, suitable for a technical team and a LinkedIn explanation. It explains our MuJoCo fly project and exactly where the latest PID video fits. Make it exceptionally legible, with a clear three-band reading order, generous spacing, restrained icons, accurate arrows, and a tasteful small fly illustration. No decorative text or illegible microcopy.

Style: precision engineering editorial graphic, charcoal #1F1F1F background, panels #2B2D2E, warm white #E6E1DB text, amber #FFC31F and machinery orange #E88107 accents, muted green #70A88A for validated status. Thin clean connector lines, crisp sans-serif typography, subtle grid. A stylized connectome illustration in the central brain box should look like a delicate neural graph, not a human brain. It is a schematic, not recorded neural activity.

Exact main title:
"FROM FLIGHT TEACHER TO FLY BRAIN"
Subtitle:
"Validate the body. Learn motor control. Add decisions through the same core."

BAND 1 — current, completed reference, green status badge:
"NOW · PID + PHYSICS VALIDATED"
Four boxes, left to right, connected with solid arrows:
"Velocity + turn commands" → "PID teacher" → "Wing-joint targets" → "MuJoCo fly"
Inside or beneath the MuJoCo fly box:
"Measured wing motion → flight forces"
A return feedback arrow FROM the MuJoCo fly TO the PID teacher, labeled:
"Body + wing feedback"
A compact status badge reads:
"Latest video · 50/50 checks · 1× playback"
This band must clearly represent the current video, with a PID controlling the physical fly, not a trained brain.

BAND 2 — the student architecture, next to train, larger than band 1:
"NEXT · TRAIN THE MOTOR STUDENT"
Left-to-right boxes connected with arrows:
"Commands + sensors" → "Encoder" → "MaleCNS recurrent core" → "Motor readout" → "Same MuJoCo fly"
Encoder label: "TRAINABLE"
Inside the central MaleCNS box, distinctly label both:
"Wiring: FIXED"
"Cell dynamics: TRAINABLE"
Draw a small recurrence loop on the MaleCNS core to indicate neural memory.
Motor readout label: "TRAINABLE"
Under the final fly box: "Same body + flight model"
Draw a feedback arrow from the final fly back to Commands + sensors, labeled "Measured state".
A dashed amber supervision connector from the teacher's wing-joint targets in band 1 to the student's motor-output connection in band 2 is labeled:
"Imitation: match teacher actions"
This connector supplies training labels, not an extra runtime command. Do not draw any sensor-to-motor shortcut around the MaleCNS core.
A readable note under this architecture:
"Fresh student training has not started."

BAND 3 — future training roadmap, muted cards connected left to right, clearly marked:
"PATH TO THE FULL FLY"
Four cards:
"1 · Imitation + recovery"
"Learn teacher actions and corrections"
then
"2 · PPO refinement"
"Simulator rewards · training-only critic"
then
"3 · Needs + utility"
"Decisions through the SAME core"
then
"4 · Multi-fly world"
"Food · water · swatter · hot light"

Bottom takeaway, readable:
"The teacher guides learning. The student must fly alone at evaluation."

Accuracy constraints:
The graph wiring is fixed, but neuron dynamics and input/output interfaces can learn. Never label the entire brain frozen. The PID is the current reference and future training teacher; it is not part of the deployed learned motor policy. PPO and utility are future stages for this fresh student. Do not imply these are already trained. Do not add an external utility network or a pose/velocity planning network outside the core. Do not show the critic issuing actions or creating rewards. The physics model uses measured wing motion to generate forces, not direct commands to body position. Keep all exact labels correctly spelled. Use straightforward rectangular cards and non-crossing arrows. The diagram itself is the image; no mockup frame, watermark, logos, or surrounding desk scene.

## Correction prompt

Edit this architecture infographic, preserving its layout, all three bands, main boxes, fly illustrations, core graph, exact existing technical labels, and the roadmap. Make only these accuracy and readability fixes:
1. Remove the current dashed 'Imitation: match teacher actions' connector that incorrectly originates from the top of the MaleCNS core. It must NOT originate from the core. Draw the dashed amber supervision connector starting at the BOTTOM edge of the 'Wing-joint targets' box in the TOP band, continuing down into the middle band, then ending at the TOP of the 'Motor readout' box. Keep the label 'Imitation: match teacher actions'. If it crosses the body's feedback line, use a clear bridge/jump or small gap so the crossing is visibly NOT a junction. This is a training-label connection from the PID teacher's output to student supervision, not another runtime path.
2. Replace the washed-out bright fog at the top and bottom with a consistent dark charcoal #1F1F1F background. Make the title and footer fully filled warm white, strong and readable. No glowing haze over typography.
3. Remove the extra decorative slogans 'SAME BODY. RICHER BEHAVIOR AHEAD.' and 'SAME CORE. MORE LIFE AHEAD.' Keep the main title, subtitle and bottom teaching/evaluation sentence.
4. In 'Commands + sensors', replace the eye icon with a simple mechanical joint icon, since the present motor student uses state sensors rather than camera images.
Everything else stays the same. Preserve 'Wiring: FIXED' AND 'Cell dynamics: TRAINABLE', the training-not-started note, all four future roadmap cards, and the 50/50 1x video badge. Crisp professional flat engineering infographic.
