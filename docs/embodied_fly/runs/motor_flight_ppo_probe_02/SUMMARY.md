# Physical-reward continuation fails flight

Source `13060c9`, parent motor_decoder_01, seed 59001. RTX 4090 neural
learning with 32 native MuJoCo CPU/mjbatch worlds. Setup 5.615176 s;
training 60.647209 s, comprising collection 30.151184 s, PPO optimization
24.881273 s and ground rehearsal 5.598249 s. 49,152 physical transitions
provide 9.8304 aggregate simulated seconds; 12,288 rehearsal frames are
counted separately. Peak CUDA allocation 18,546,130,432 bytes.

All 750 completed episodes failed within 1.6–34 ms. Final-state replay
classifies 544 as low upright, 204 as low height and two as both; this is
post-action state and may differ by one physics step from termination.
All failure traces were hash/finite-state verified on the GPU machine.
Both independent hover/forward tests fail (RMSE 11.455/16.872 mm). Five
fixed ground cases remain stable, with hold falling. Continuous walk/stop/
resume stays upright; stop fails with 0.926 mm late displacement. All
evaluations have zero warnings. No checkpoint promotion.

This result preceded the user-approved course correction to flight dynamics
driven by measured wing motion. The original aerodynamic trial is preserved.
