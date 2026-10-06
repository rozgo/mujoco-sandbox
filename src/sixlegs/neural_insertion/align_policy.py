"""Evaluate a PufferLib alignment checkpoint and render recorded episodes.

Evaluation runs the checkpoint with Puffer's own CPU network code (deterministic
Gaussian mean) on the shared C core, one fresh world per predetermined seed,
through experiments/neural_insertion/puffer/align_eval.c. Video replays the
recorded 50 Hz joint states in the full scene; pixels are observer output.
"""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw

from .align_env import EPISODE_FIELDS, EVALUATION_SEEDS
from .motion import JOINT_NAMES
from .motion_media import ACCENT, BG, INK, MUTED, WARN, H, W, caption, follow_camera
from .rl_scene import RL_SCENE, build_rl_scene
from .scene import ROOT, load_scene
from .visuals import font, options

EVALUATOR = ROOT/"build/neural_insertion/align_eval"
PUFFER = ROOT/"build/neural_insertion/pufferlib"
PUFFER_REVISION = "6ffa5b10dbbbe4d1e8288367c7d9d3acd3bad4a2"
FPS = 30


class Policy:
    """A trained checkpoint run by PufferLib's own CPU network code (deterministic mean)."""

    def __init__(self, weights, hidden=128, layers=2):
        import ctypes
        import platform
        source = ROOT/"experiments/neural_insertion/puffer/align_policy_lib.c"
        revision = subprocess.check_output(["git", "-C", str(PUFFER), "rev-parse", "HEAD"], text=True).strip()
        if revision != PUFFER_REVISION:
            raise RuntimeError("PufferLib checkout must be at the pinned revision; run puffer/install.sh")
        key = hashlib.sha256(source.read_bytes()+(PUFFER/"src/puffercpu.c").read_bytes()).hexdigest()[:16]
        lib = ROOT/f"build/neural_insertion/align_policy/{key}/libpolicy.{'dylib' if platform.system() == 'Darwin' else 'so'}"
        if not lib.exists():
            lib.parent.mkdir(parents=True, exist_ok=True)
            raylib = next(PUFFER.glob("raylib-5.5_*"))
            subprocess.run(["clang", "-O2", "-std=gnu11", "-w", "-shared", "-fPIC", f"-I{PUFFER/'src'}",
                            f"-I{PUFFER/'vendor'}", f"-I{raylib/'include'}", str(source), "-lm", "-o", str(lib)], check=True)
        self.lib = ctypes.CDLL(str(lib))
        f32 = np.ctypeslib.ndpointer(np.float32, flags="C")
        self.lib.policy_create.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self.lib.policy_create.restype = ctypes.c_void_p
        self.lib.policy_act.argtypes = [ctypes.c_void_p, f32, ctypes.c_float, f32]
        self.handle = self.lib.policy_create(str(weights).encode(), 16, hidden, layers, 3)
        if not self.handle:
            raise ValueError("Checkpoint does not match the alignment network")
        self.weights_sha256 = hashlib.sha256(Path(weights).read_bytes()).hexdigest()

    def act(self, obs, first):
        action = np.zeros(3, np.float32)
        self.lib.policy_act(self.handle, np.ascontiguousarray(obs, np.float32), 1. if first else 0., action)
        return action


def evaluate_checkpoint(weights, output, hidden=128, layers=2, record=()):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    xml, _ = build_rl_scene(RL_SCENE)
    started = time.perf_counter()
    subprocess.run([str(EVALUATOR), str(xml), str(weights), str(hidden), str(layers), str(EVALUATION_SEEDS[0]),
                    str(len(EVALUATION_SEEDS)), str(output), *map(str, record)], check=True)
    episodes = json.loads((output/"episodes.json").read_text())
    summary = {k: float(np.mean([e[k] for e in episodes])) for k in EPISODE_FIELDS}
    good = [e for e in episodes if e["success"]]
    summary.update(episodes=len(episodes), wall_s=time.perf_counter()-started,
                   mean_success_time_s=float(np.mean([e["episode_length"] for e in good])*.02) if good else None,
                   max_success_lateral_um=float(max(e["final_lateral_um"] for e in good)) if good else None,
                   max_success_vertical_um=float(max(e["final_vertical_um"] for e in good)) if good else None,
                   weights_sha256=hashlib.sha256(Path(weights).read_bytes()).hexdigest(),
                   model_sha256=hashlib.sha256(xml.read_bytes()).hexdigest(),
                   evaluator="Puffer CPU PufferNet, deterministic mean action")
    (output/"summary.json").write_text(json.dumps(summary, indent=1)+"\n")
    return summary, episodes


def load_states(folder, seed):
    rows = np.fromfile(Path(folder)/f"states_{seed}.bin", dtype=np.float64).reshape(-1, 14)
    return {"tip": rows[:, :3], "goal": rows[:, 3:6], "qpos": rows[:, 6:11], "time": rows[:, 11],
            "target": int(rows[0, 12]), "action_norm": rows[:, 13]}


def panel(states, i, seed, episode, index, total, baseline):
    image = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(image)
    t = states["time"][i]
    tip, goal = states["tip"][i], states["goal"][i]
    lateral = np.linalg.norm(tip[:2]-goal[:2])*1e6
    vertical = (tip[2]-goal[2])*1e6
    done = i == len(states["time"])-1
    draw.text((28, 22), "LEARNED ALIGNMENT POLICY  /  PUFFERLIB 5.0", font=font(24), fill=INK)
    draw.text((28, 58), "Alignment only: hover 1 mm above the target. No thread, no insertion. Real time 1.0x",
              font=font(16), fill=MUTED)
    draw.text((28, 100), f"Episode {index+1} of {total}   seed {seed}   target {states['target']}", font=font(22), fill=INK)
    draw.text((28, 140), f"t = {t:5.2f} s", font=font(34), fill=INK)
    status = ("SUCCESS" if episode["success"] else "COLLISION" if episode["collision"] else "TIMEOUT") if done \
        else ("HOLDING" if lateral < 10 and abs(vertical) < 10 else "MOVING")
    draw.text((330, 148), status, font=font(26), fill=ACCENT if status in ("SUCCESS", "HOLDING") else WARN)
    rows = [("Lateral error to hover point", f"{lateral:10.1f} µm"),
            ("Vertical error to hover point", f"{vertical:10.1f} µm"),
            ("Success tolerance", "10 µm, held 0.3 s"),
            ("Action magnitude", f"{states['action_norm'][i]:10.3f}")]
    for k, (label, value) in enumerate(rows):
        y = 210+k*42
        draw.text((28, y), label, font=font(19), fill=MUTED)
        draw.text((610, y), value, font=font(21), fill=INK)
    draw.text((28, 400), "Inputs are exact simulator states, not camera pixels.", font=font(16), fill=MUTED)
    draw.text((28, 430), f"Scripted reference on this seed: {baseline:.2f} s to success", font=font(16), fill=MUTED)
    return image


def results_card(summary, scripted):
    card = Image.new("RGB", (1920, 1080), BG)
    draw = ImageDraw.Draw(card)
    draw.text((90, 80), "RESULTS  /  LEARNED ALIGNMENT, 200 PREDETERMINED SEEDS", font=font(44), fill=INK)
    draw.text((90, 148), "Robot-only alignment, no thread  /  PufferLib 5.0 on RTX 4090  /  evaluated on CPU, deterministic actions",
              font=font(22), fill=MUTED)
    rows = [("", "Learned policy", "Scripted reference"),
            ("Success", f"{100*summary['success']:.1f}%", f"{100*scripted['success']:.1f}%"),
            ("Collision", f"{100*summary['collision']:.1f}%", f"{100*scripted['collision']:.1f}%"),
            ("Timeout", f"{100*summary['timeout']:.1f}%", f"{100*scripted['timeout']:.1f}%"),
            ("Mean time to success", f"{summary['mean_success_time_s'] or float('nan'):.2f} s",
             f"{scripted['mean_success_time_s']:.2f} s"),
            ("Worst final lateral error, successes", f"{summary['max_success_lateral_um'] or float('nan'):.2f} µm",
             f"{scripted['max_success_lateral_um']:.2f} µm")]
    for k, (label, a, b) in enumerate(rows):
        y = 270+k*90
        style = font(30) if k else font(26)
        draw.text((90, y), label, font=style, fill=MUTED)
        draw.text((880, y), a, font=style, fill=ACCENT if k == 0 else INK)
        draw.text((1380, y), b, font=style, fill=MUTED if k == 0 else INK)
    return card


def record(folder, seeds, summary, scripted_episodes, scripted_summary, output, card_s=3.):
    started = time.perf_counter()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    video = output/"learned_alignment.mp4"
    episodes = {e["seed"]: e for e in json.loads((Path(folder)/"episodes.json").read_text())}
    scripted = {e["seed"]: e for e in scripted_episodes}
    model, data = load_scene()
    qadr = [model.jnt_qposadr[model.joint(n).id] for n in JOINT_NAMES]
    writer = imageio_ffmpeg.write_frames(str(video), (1920, 1080), fps=FPS, codec="libx264", quality=8,
                                         macro_block_size=2, output_params=["-movflags", "+faststart"])
    writer.send(None)
    stills, frames = {}, 0
    with mujoco.Renderer(model, H, W) as renderer:
        opt = options()
        for index, seed in enumerate(seeds):
            states = load_states(folder, seed)
            for t in np.arange(0, states["time"][-1]+1e-9, 1/FPS):
                i = int(np.argmin(abs(states["time"]-t)))
                data.qpos[qadr] = states["qpos"][i]  # replay of a recorded state
                mujoco.mj_forward(model, data)
                frame = Image.new("RGB", (1920, 1080), BG)
                renderer.update_scene(data, camera="overview", scene_option=opt)
                frame.paste(caption(Image.fromarray(renderer.render()), "OVERVIEW", "fixed world camera"), (0, 0))
                renderer.update_scene(data, camera=follow_camera(data.site("needle_tip").xpos.copy()), scene_option=opt)
                frame.paste(caption(Image.fromarray(renderer.render()), "NEEDLE CLOSE-UP", "observer camera following the tip"), (W, 0))
                renderer.update_scene(data, camera="microscope", scene_option=opt)
                frame.paste(caption(Image.fromarray(renderer.render()), "MOUNTED MICROSCOPE", "observer only, not a policy input"), (0, H))
                frame.paste(panel(states, i, seed, episodes[seed], index, len(seeds),
                                  scripted[seed]["episode_length"]*.02), (W, H))
                writer.send(np.asarray(frame))
                if i in (0, len(states["time"])-1) or (abs(t-1.) < .5/FPS):
                    stills[f"seed{seed}_{t:05.2f}s"] = frame
                frames += 1
        card = results_card(summary, scripted_summary)
        for _ in range(int(card_s*FPS)):
            writer.send(np.asarray(card))
    writer.close()
    for name, frame in stills.items():
        frame.save(output/f"{name}.png")
    card.save(output/"results_card.png")
    receipt = {"video": video.name, "video_sha256": hashlib.sha256(video.read_bytes()).hexdigest(), "fps": FPS,
               "frames": frames+int(card_s*FPS), "seeds": list(seeds), "playback": "1.0x real time",
               "render_wall_s": time.perf_counter()-started, "additional_simulation_s": 0,
               "state_rate_hz": 50, "frame_selection": "nearest recorded 50 Hz state"}
    (output/"video_manifest.json").write_text(json.dumps(receipt, indent=2)+"\n")
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("weights", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baselines", type=Path, default=ROOT/"docs/neural_insertion/ALIGN_BASELINES.json")
    parser.add_argument("--video", type=Path)
    parser.add_argument("--record", type=int, nargs="*", default=[])
    args = parser.parse_args()
    summary, episodes = evaluate_checkpoint(args.weights, args.output, record=args.record)
    print(json.dumps(summary, indent=1))
    if args.video and args.record:
        baselines = json.loads(args.baselines.read_text())["policies"]["scripted_reference"]
        print(json.dumps(record(args.output, args.record, summary, baselines["episodes"], baselines["summary"],
                                args.video), indent=1))


if __name__ == "__main__":
    main()
