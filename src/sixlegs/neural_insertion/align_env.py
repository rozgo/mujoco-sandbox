"""Local build, wrapper and evaluation for the native alignment environment.

The same C core (native/surgical_core.h) is compiled into the PufferLib 5.0
adapter for training and into this ctypes library for tests, baselines and
policy evaluation on CPU MuJoCo.
"""

import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time

import mujoco
import numpy as np

from .rl_scene import RL_SCENE, build_rl_scene
from .scene import ROOT

NATIVE = Path(__file__).with_name("native")
OBS, ACT = 16, 3
ROW = 40  # surgical_core.h SA_ROW
EPISODE_FIELDS = ("success", "collision", "timeout", "vessel_steps", "final_lateral_um", "final_vertical_um",
                  "episode_return", "episode_length", "level")
# Predetermined, never used for training or tuning.
EVALUATION_SEEDS = tuple(range(1_000_000, 1_000_200))
_lib = None


def library():
    global _lib
    if _lib is not None:
        return _lib
    sources = [NATIVE/"surgical_core.h", NATIVE/"align_ctypes.c"]
    key = hashlib.sha256(b"".join(p.read_bytes() for p in sources)+mujoco.__version__.encode()
                         + platform.platform().encode()).hexdigest()[:16]
    folder = ROOT/"build/neural_insertion/align_native"/key
    folder.mkdir(parents=True, exist_ok=True)
    darwin = platform.system() == "Darwin"
    dest = folder/("libalign.dylib" if darwin else "libalign.so")
    if not dest.exists():
        mjdir = Path(mujoco.__file__).parent
        libs = list(mjdir.glob("libmujoco.*.dylib" if darwin else "libmujoco.so.*"))
        if len(libs) != 1:
            raise RuntimeError("Expected one installed MuJoCo library")
        subprocess.run(["clang" if darwin else "cc", "-O2", "-std=c11", "-shared", "-fPIC", "-Wall", "-Werror",
                        "-I"+str(mjdir/"include"), "-I"+str(NATIVE), str(NATIVE/"align_ctypes.c"), str(libs[0]),
                        "-Wl,-rpath,"+str(mjdir), "-lm", "-o", str(dest)], check=True)
    lib = ctypes.CDLL(str(dest))
    f32 = np.ctypeslib.ndpointer(np.float32, flags="C")
    f64 = np.ctypeslib.ndpointer(np.float64, flags="C")
    lib.sa_create.argtypes, lib.sa_create.restype = [ctypes.c_char_p, ctypes.c_ulonglong], ctypes.c_void_p
    lib.sa_disturbance_c.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_int]
    lib.sa_destroy.argtypes = [ctypes.c_void_p]
    lib.sa_reset_c.argtypes = [ctypes.c_void_p, f32]
    lib.sa_step_c.argtypes, lib.sa_step_c.restype = [ctypes.c_void_p, f32, f32, f32, f32], ctypes.c_int
    lib.sa_scripted_c.argtypes = [f32, f32]
    lib.sa_state.argtypes = [ctypes.c_void_p, f64]
    lib.sa_rollout_scripted.argtypes = [ctypes.c_void_p, ctypes.c_int, f32]
    lib.sa_replay_row_c.argtypes = [ctypes.c_void_p, ctypes.c_double, f64]
    _lib = lib
    return lib


class AlignEnv:
    """One native world. Observations and rewards come from the shared C core."""

    def __init__(self, seed, xml=RL_SCENE, level=0., mode=0):
        """level in [0, 1] scales the disturbance layer; mode 1 draws each episode's level in [0, level]."""
        if not Path(xml).exists():
            build_rl_scene(xml)
        self.lib = library()
        self.handle = self.lib.sa_create(str(xml).encode(), seed)
        self.lib.sa_disturbance_c(self.handle, level, mode)
        self.obs = np.zeros(OBS, np.float32)
        self.reward = np.zeros(1, np.float32)
        self.episode = np.zeros(len(EPISODE_FIELDS), np.float32)

    def reset(self):
        self.lib.sa_reset_c(self.handle, self.obs)
        return self.obs.copy()

    def step(self, action):
        action = np.ascontiguousarray(action, np.float32)
        done = self.lib.sa_step_c(self.handle, action, self.obs, self.reward, self.episode)
        info = dict(zip(EPISODE_FIELDS, self.episode.tolist())) if done else None
        return self.obs.copy(), float(self.reward[0]), bool(done), info

    def state(self):
        out = np.zeros(18)
        self.lib.sa_state(self.handle, out)
        return {"tip": out[:3], "goal": out[3:6], "qpos": out[6:11], "q_ref": out[11:16],
                "target": int(out[16]), "tick": int(out[17])}

    def replay_row(self, action_norm=0.):
        """Replay row (surgical_core.h SA_ROW): true and measured state plus every disturbance component."""
        out = np.zeros(ROW)
        self.lib.sa_replay_row_c(self.handle, action_norm, out)
        return out

    def scripted_action(self, obs):
        action = np.zeros(ACT, np.float32)
        self.lib.sa_scripted_c(np.ascontiguousarray(obs, np.float32), action)
        return action

    def close(self):
        if self.handle:
            self.lib.sa_destroy(self.handle)
            self.handle = None

    def __del__(self):
        self.close()


def evaluate(policy, seeds=EVALUATION_SEEDS, record=None, level=0.):
    """One episode per predetermined seed at a fixed disturbance level. `policy(env, obs)` returns an action."""
    results, started = [], time.perf_counter()
    for seed in seeds:
        env = AlignEnv(seed, level=level)
        obs, done, states = env.reset(), False, [env.state()]
        while not done:
            obs, _, done, info = env.step(policy(env, obs))
            if record is not None and seed in record:
                states.append(env.state())
        info["seed"], info["target"] = seed, states[0]["target"]
        results.append(info)
        if record is not None and seed in record:
            record[seed] = states
        env.close()
    summary = {k: float(np.mean([r[k] for r in results])) for k in EPISODE_FIELDS}
    successes = [r for r in results if r["success"]]
    summary.update(episodes=len(results), wall_s=time.perf_counter()-started,
                   mean_success_time_s=float(np.mean([r["episode_length"] for r in successes])*.02) if successes else None,
                   max_success_lateral_um=float(max(r["final_lateral_um"] for r in successes)) if successes else None)
    return summary, results


def record_episode(policy, seed, level=0.):
    """One episode with a replay row per policy step, as align_eval records for checkpoints."""
    env = AlignEnv(seed, level=level)
    obs, done, norm, rows = env.reset(), False, 0., []
    while not done:
        rows.append(env.replay_row(norm))
        action = np.asarray(policy(env, obs), np.float32)
        norm = float(np.linalg.norm(action))
        obs, _, done, info = env.step(action)
    rows.append(env.replay_row(norm))
    info["seed"], info["target"] = seed, int(rows[0][12])
    env.close()
    return info, np.array(rows)


def scripted(env, obs):
    return env.scripted_action(obs)


def zero(env, obs):
    return np.zeros(ACT, np.float32)


def random_policy(rng):
    return lambda env, obs: rng.uniform(-1, 1, ACT).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("baselines",))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    path, meta = build_rl_scene()
    report = {"started_utc": datetime.now(timezone.utc).isoformat(), "mujoco": mujoco.__version__,
              "scene_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "core_sha256": hashlib.sha256((NATIVE/"surgical_core.h").read_bytes()).hexdigest(),
              "evaluation_seeds": [EVALUATION_SEEDS[0], EVALUATION_SEEDS[-1]], "policies": {}}
    for name, policy in (("scripted_reference", scripted), ("zero_action", zero),
                         ("uniform_random", random_policy(np.random.default_rng(7)))):
        summary, results = evaluate(policy)
        report["policies"][name] = {"summary": summary, "episodes": results}
        print(name, json.dumps(summary), flush=True)
    (args.output/"manifest.json").write_text(json.dumps(report, indent=1)+"\n")


if __name__ == "__main__":
    main()
