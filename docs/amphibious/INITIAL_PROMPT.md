# Amphibious attachment demo brief

Reference shared by the user: [Chris Klaftenegger's LinkedIn video](https://www.linkedin.com/posts/chris-klaftenegger_onyxindustries-sentry-raw15-ugcPost-7500971822365081600-1sk5/).

> What is the minimum sim we can do that shows this in mujoco so that we can create a video showing our sim. It does not need radio/comms, nor does it need to be autonomous... we just need to show navigation into and out of whatever and how the attachments work. What do you think?

> Perfect, lets add a bit of uneven roughness comming in and out, to show that its really navigating over terrain. what do you think? some rock like shapes, or something.

> Perfect, lets make a new sim for this. Make this a new goal. From sim to full video captured. Start goal and implementation now.

Subsequent scope correction, which supersedes the roughness request:

> lets remove the rocks, and favor what we are trully trying to show here

Further clarification:

> are we using walking dynamics already programmed for this robot? or we are hacking it through? and is the dog extending its front legs like in the video, such that the floats are pointing forward along with the legs?

Implementation scope: a separate, unarmed Go2 reference quadruped with four rigid leg-mounted flotation capsules; smooth bank entry, floating passage, and bank exit. Front legs extend forward in the floating posture. Scripted control is acceptable; no communications or autonomous perception. This is an illustrative attachment concept, not a calibrated reproduction of the reference hardware. Preserve the previous hexapod and rover demos. Use uv and Git LFS, test on macOS, capture a complete multi-view video, commit the result, and record elapsed time.
