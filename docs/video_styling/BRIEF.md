# Cinematic presentation trial

Restyle a short segment of the accepted MuJoCo robot footage through the official
WaveSpeed Python SDK. Give the robot brushed-metal surfaces and improve the
environment and lighting. Research current video-editing models, generate one
small trial, then compare the result with the exact source clip.

Keep credentials in the ignored local `.env`; public examples contain placeholders
only. Preserve accepted simulations, model weights and original videos. Work on
`feature/cinematic-video-styling`, after the accepted moving-support release on main.

First input: five seconds of the healthy robot on the moving platform, from the
single third-person camera in the accepted comparison film. Crop away the dashboard
and other cameras. Keep the same camera and timing during the requested edit.

Review material appearance, environment quality, temporal flicker, platform motion,
leg count and geometry, foot contact and timing. Decode every output frame and make
a labeled source/generated comparison. Generative output is presentation footage;
it cannot replace the original measured physics or establish new controller results.
