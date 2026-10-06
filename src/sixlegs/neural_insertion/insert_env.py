"""Local build, wrapper and evaluation for the native thread-insertion environment (native/insert_core.h).

The same C core is compiled into the PufferLib 5.0 adapter for training and into this ctypes library for
tests, yardsticks and policy evaluation on CPU MuJoCo. One site per episode; see insert_core.h.
"""

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import platform
import subprocess

import mujoco
import numpy as np

from .rl_scene import RL_SCENE, build_rl_scene
from .scene import ROOT

NATIVE = Path(__file__).with_name("native")
OBS, ACT = 24, 4
EPISODE_FIELDS = ("success", "collision", "thread_touch", "dragged", "missed", "timeout", "placement_um", "depth_um",
                  "vessel_steps", "episode_return", "episode_length", "level", "nplaced")
# Predetermined evaluation seeds, never used for training or tuning (each draws site, rotation, placed threads,
# thread-end offset and jump, start pose and disturbances).
EVALUATION_SEEDS = tuple(range(2_000_000, 2_000_200))
_lib = None


def library():
    global _lib
    if _lib is not None:
        return _lib
    sources = [NATIVE/"surgical_core.h", NATIVE/"insert_core.h", NATIVE/"insert_core_stats.h", NATIVE/"insert_ctypes.c"]
    key = hashlib.sha256(b"".join(p.read_bytes() for p in sources)+mujoco.__version__.encode()
                         + platform.platform().encode()).hexdigest()[:16]
    folder = ROOT/"build/neural_insertion/insert_native"/key
    folder.mkdir(parents=True, exist_ok=True)
    darwin = platform.system() == "Darwin"
    dest = folder/("libinsert.dylib" if darwin else "libinsert.so")
    if not dest.exists():
        mjdir = Path(mujoco.__file__).parent
        libs = list(mjdir.glob("libmujoco.*.dylib" if darwin else "libmujoco.so.*"))
        subprocess.run(["clang" if darwin else "cc", "-O2", "-std=c11", "-shared", "-fPIC", "-Wall", "-Werror",
                        "-I"+str(mjdir/"include"), "-I"+str(NATIVE), str(NATIVE/"insert_ctypes.c"), str(libs[0]),
                        "-Wl,-rpath,"+str(mjdir), "-lm", "-o", str(dest)], check=True)
    lib = ctypes.CDLL(str(dest))
    f32 = np.ctypeslib.ndpointer(np.float32, flags="C")
    f64 = np.ctypeslib.ndpointer(np.float64, flags="C")
    lib.si_create.argtypes, lib.si_create.restype = [ctypes.c_char_p, ctypes.c_ulonglong], ctypes.c_void_p
    lib.si_disturbance_c.argtypes = [ctypes.c_void_p, ctypes.c_double, ctypes.c_int]
    lib.si_destroy.argtypes = [ctypes.c_void_p]
    lib.si_reset_c.argtypes = [ctypes.c_void_p, f32]
    lib.si_step_c.argtypes, lib.si_step_c.restype = [ctypes.c_void_p, f32, f32, f32, f32], ctypes.c_int
    lib.si_scripted_c.argtypes = [f32, f32, ctypes.c_int]
    lib.si_state.argtypes = [ctypes.c_void_p, f64]
    _lib = lib
    return lib


class InsertEnv:
    """One native world. Observations and rewards come from the shared C core."""

    def __init__(self, seed, xml=RL_SCENE, level=0., mode=0):
        if not Path(xml).exists():
            build_rl_scene(xml)
        self.lib = library()
        self.handle = self.lib.si_create(str(xml).encode(), seed)
        self.lib.si_disturbance_c(self.handle, level, mode)
        self.obs = np.zeros(OBS, np.float32)
        self.reward = np.zeros(1, np.float32)
        self.episode = np.zeros(len(EPISODE_FIELDS), np.float32)

    def reset(self):
        self.lib.si_reset_c(self.handle, self.obs)
        return self.obs.copy()

    def step(self, action):
        a = np.ascontiguousarray(action, np.float32)
        done = self.lib.si_step_c(self.handle, a, self.obs, self.reward, self.episode)
        info = dict(zip(EPISODE_FIELDS, self.episode.tolist())) if done else None
        return self.obs.copy(), float(self.reward[0]), bool(done), info

    def state(self):
        out = np.zeros(16)
        self.lib.si_state(self.handle, out)
        return out

    def close(self):
        if self.handle:
            self.lib.si_destroy(self.handle)
            self.handle = None

    def __del__(self):
        self.close()


def yardstick(compensate):
    def act(env, obs):
        a = np.zeros(ACT, np.float32)
        env.lib.si_scripted_c(np.ascontiguousarray(obs, np.float32), a, int(compensate))
        return a
    return act


def evaluate(policy, seeds=EVALUATION_SEEDS, level=0.):
    infos = []
    for seed in seeds:
        env = InsertEnv(seed, level=level)
        obs, done = env.reset(), False
        while not done:
            obs, _, done, info = env.step(policy(env, obs))
        infos.append(info)
        env.close()
    return infos


def summary(infos):
    done = [i for i in infos if i["timeout"] == 0 and i["collision"] == 0 and i["thread_touch"] == 0
            and i["dragged"] == 0 and i["missed"] == 0]
    place = np.array([i["placement_um"] for i in done])
    return {"episodes": len(infos), "success": float(np.mean([i["success"] for i in infos])),
            "placed": len(done), "placement_um_median": float(np.median(place)) if len(place) else None,
            "placement_um_p90": float(np.percentile(place, 90)) if len(place) else None,
            **{k: float(np.mean([i[k] for i in infos])) for k in ("collision", "thread_touch", "dragged", "missed", "timeout")},
            "mean_length_s": float(np.mean([i["episode_length"] for i in infos]))*.02}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    out = {}
    for name, comp in (("yardstick", 0), ("compensating", 1)):
        for level in (0., 1., 2.):
            out[f"{name}_level{level:g}"] = s = summary(evaluate(yardstick(comp), EVALUATION_SEEDS[:args.episodes], level))
            print(name, level, json.dumps(s), flush=True)
    if args.output:
        args.output.write_text(json.dumps(out, indent=1)+"\n")


if __name__ == "__main__":
    main()
