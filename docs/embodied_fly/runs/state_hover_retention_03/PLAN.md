# Student-only physical collection for hover

Continue from state_hover_retention02, SHA-256
`f5d383485f37478b02b74cf9502c603f35b24e2f08b35e0ca6035b76b0bd87c6`.
Change hover collection assistance from80%to0%. The actor now executes every
action in every world. State-based teaching commands remain loss targets only,
so learning sees the learner's own hover-start and recovery states. This is
online imitation with student-only collection, not PPO.

Retain32worlds(11stand/11walk/10hover),16CPU threads,5kHz physics/500Hz control,
32-step chunks, fresh Adam1e-6,180-second allowance,2-second episode limit,
seed93001,task-loss weights4:4:1,ground wing weight10 andhover wing weight2.
The frozen parent reference now copies02 and maintains independent recurrent
state. Body, sensor schema, graph, reference gains and deployed architecture
remain unchanged. One checkpoint handles all three commands.

Validate with the same complete five-second unassisted review at seed72001.
Retain failures, measure actual timing, render and open all commands together.
