# Complete-actor wing-motion imitation

Source `d4ac5e7`, seed 63001, decoder01 parent. Same measured graph and 2,410,668-parameter actor; sensory interfaces, intrinsic cell parameters, utility and all 78 motor outputs train together. Flight labels include folded-leg posture; ground data retain walking. A phase-zero cold start removes conflicting initial reference labels without supplying phase to the actor.

RTX 4090 optimization: **180.274561 seconds**, setup 4.236592 seconds, validation 0.443605 seconds. Sixteen recurrent sequences, 32 supervised steps plus 16 burn-in at 500 Hz; 393 updates / 201,216 reused examples / 14,000 distinct training frames. No live physics worlds during fitting; peak CUDA allocation 5,200,346,624 bytes. Held-out motor MSE improved 0.103644 → 0.020944.

Both independent two-second flights failed. The hover first leaves the height/orientation envelope at 0.490 seconds, versus 0.066 seconds for the preserved parent. This is transient progress: it rises from 2 cm to over 5 cm, then falls. It is not stable hover. Five of six original-model ground cases remain stable; only one of six new-preset ground cases does. All raw ground tracking gates fail. The old walking reference remains preserved; no promotion.

The complete [student diagnostic](../../../../previews/embodied_fly/wing_motion_student_diagnostic_v1.mp4) includes the fall, actual simulated neural activity and observer eye views. Teacher-free evaluation uses all actor outputs, without a runtime wing oscillator. Capture hashes, finite state, bounded action and exact causal previous-action feedback were verified locally.
