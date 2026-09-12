# Flight curriculum: required physical setup

Flight remains part of the approved goal, inside the same learned actor. It is
not trained yet. This check records what the pinned FlyBody code actually needs
before collecting wing-control demonstrations or applying flight rewards.

The current full walking model retains six wing joints and all wing actuators.
Its compiled air density is 0.00128 g/cm³ (1.28 kg/m³), viscosity 0.000185
 g/(cm·s) (1.85e-5 Pa·s), and gravity −981 cm/s². However, its two dedicated
wing-fluid geoms currently have zero fluid-model coefficients. Their presence
and visible wing meshes do not establish an operational flight model.

The pinned upstream `Flying` task enables ellipsoid wing aerodynamics with
coefficients `[1.0, 0.5, 1.5, 1.7, 1.0]`, changes all three wing-axis gains to 18,
and uses wing damping 0.007769230 and stiffness 0.01 in its CGS convention.
The walking composition currently uses gains 3/2/1 and damping 0.0005.
Upstream flight uses 20 kHz physics and 5 kHz control, compared with our current
5 kHz / 500 Hz terrestrial curriculum. Its reference wingbeat is 218 Hz.
These are source settings, not locally validated flight results.

Before training flight:

1. Add an explicit flight-ready physical preset, preserve current walking
   reproduction, and measure lift/drag, wing torque saturation and timestep
   convergence with the complete body. Keep meaningful floor contacts and legs;
   upstream flight task defaults disable both, which is unsuitable for takeoff
   and landing in our arena.
2. Validate ground-to-air and air-to-ground contacts. Lift must come from wing
   motion interacting with the fluid model, without root forces or pose writes.
3. Resolve the control-rate change explicitly. Preserve one controller/checkpoint
   and demonstrate that its recurrent timing and terrestrial skills survive the
   higher wing-control rate or a documented internal multirate implementation.
   Merely calling existing walking weights ten times faster is not validation.
4. If using the official flight teacher for training data, capture its complete
   executed wing controls. Its `FlightImitationWBPG` combines a learned residual
   with an external wingbeat generator. That may supply demonstrations, but the
   student must generate its own individual wing controls with no runtime pattern
   generator, and attribution must identify the inherited motion knowledge.
5. Extend the same actor through wing control, stable free flight, takeoff,
   maneuvering and landing while retaining walking/stopping. Then train the
   survival decisions that select and combine those learned skills.

Sources inspected locally at FlyBody revision
`d015e9bfe441bd90ae431bac24c55cb74bdbce26`:

- [Full anatomy and defaults](https://github.com/TuragaLab/flybody/blob/d015e9bfe441bd90ae431bac24c55cb74bdbce26/flybody/fruitfly/assets/fruitfly.xml).
- [Flight physical setup](https://github.com/TuragaLab/flybody/blob/d015e9bfe441bd90ae431bac24c55cb74bdbce26/flybody/tasks/base.py).
- [Flight constants](https://github.com/TuragaLab/flybody/blob/d015e9bfe441bd90ae431bac24c55cb74bdbce26/flybody/tasks/constants.py).
- [Teacher plus wingbeat generator](https://github.com/TuragaLab/flybody/blob/d015e9bfe441bd90ae431bac24c55cb74bdbce26/flybody/tasks/flight_imitation.py).
