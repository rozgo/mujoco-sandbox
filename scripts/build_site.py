"""Build the project website pilot into build/site (ignored). Publishes nothing.

Steps: record evaluation episodes for each policy at each disturbance level,
export scene and replays for the raylib viewer, compile the viewer natively
(inspection) and with Emscripten (web), and assemble the journal page with
converted media. Usage: uv run --locked python scripts/build_site.py [--native-only]
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))

from sixlegs.neural_insertion.align_env import EVALUATION_SEEDS, record_episode, scripted  # noqa: E402
from sixlegs.neural_insertion.align_policy import evaluate_checkpoint, load_states, states_from_rows  # noqa: E402
from sixlegs.neural_insertion.scene import load_scene  # noqa: E402
from sixlegs.neural_insertion.site_export import write_replays, write_scene  # noqa: E402

SITE, NATIVE = ROOT/"build/site", ROOT/"build/site_native"
DEPS = ROOT/"build/site_deps"
WEIGHTS = {"robust": ROOT/"assets/neural_insertion/align_v4_policy.bin",
           "undisturbed": ROOT/"assets/neural_insertion/align_v3_policy.bin"}
STATES = ROOT/"outputs/neural_insertion/site/states"
# Viewer policies: 0 robust learned (trained under disturbances), 1 scripted
# yardstick, 2 learned without disturbances. Each at three disturbance levels on
# the first SITE_SEEDS predetermined evaluation seeds (results use all 200).
LEVELS = (0.0, 1.0, 2.0)  # disturbance scale v2: undisturbed, nominal, stress
SITE_SEEDS = EVALUATION_SEEDS[:30]
FONT = Path(__import__("matplotlib").get_data_path())/"fonts/ttf"


def outcome(info):
    return 1 if info["success"] else 2 if info["collision"] else 0


def learned_episodes(name, policy, level):
    weights = WEIGHTS[name]
    core = (ROOT/"src/sixlegs/neural_insertion/native/surgical_core.h").read_bytes()
    tag = hashlib.sha256(weights.read_bytes()+core).hexdigest()[:12]  # replays depend on weights and core
    folder = STATES/f"{name}_{tag}_level{level:.2f}"
    if not (folder/"episodes.json").exists():
        evaluate_checkpoint(weights, folder, record=SITE_SEEDS, level=level)
    infos = {e["seed"]: e for e in json.loads((folder/"episodes.json").read_text())}
    return [{"seed": seed, "policy": policy, "outcome": outcome(infos[seed]), "states": load_states(folder, seed)}
            for seed in SITE_SEEDS]


def scripted_episodes(level):
    episodes = []
    for seed in SITE_SEEDS:
        info, rows = record_episode(scripted, seed, level)
        episodes.append({"seed": seed, "policy": 1, "outcome": outcome(info), "states": states_from_rows(rows)})
    return episodes


def fonts(folder):
    """DejaVu Sans (from matplotlib; free licence, copied alongside), subset to the viewer's characters."""
    from fontTools import subset
    text = "".join(map(chr, range(32, 127)))+"\u00b5\u00b2\u00b1\u00d7\u2212"
    for source, name in (("DejaVuSans.ttf", "font.ttf"), ("DejaVuSans-Bold.ttf", "font_bold.ttf")):
        font = subset.load_font(str(FONT/source), subset.Options())
        subsetter = subset.Subsetter()
        subsetter.populate(text=text)
        subsetter.subset(font)
        font.save(str(folder/name))
    shutil.copy2(FONT/"LICENSE_DEJAVU", folder/"LICENSE_DEJAVU")


def export_data(folder):
    folder.mkdir(parents=True, exist_ok=True)
    model, data = load_scene()
    scene = write_scene(folder/"scene.bin", model, data)
    episodes = []
    for level in LEVELS:
        episodes += learned_episodes("robust", 0, level)+scripted_episodes(level)+learned_episodes("undisturbed", 2, level)
    write_replays(folder/"replays.bin", model, data, episodes)
    fonts(folder)
    summary = {"scene": scene, "episodes": len(episodes), "seeds": [SITE_SEEDS[0], SITE_SEEDS[-1]],
               "replay_bytes": (folder/"replays.bin").stat().st_size,
               "success_on_site_seeds": {f"policy{p}_level{level}": float(np.mean(
                   [e["outcome"] == 1 for e in episodes if e["policy"] == p and e["states"]["level"][0] == level]))
                   for p in (0, 1, 2) for level in LEVELS}}
    (folder/"export.json").write_text(json.dumps(summary, indent=1)+"\n")
    return summary


def build_native():
    NATIVE.mkdir(parents=True, exist_ok=True)
    raylib = DEPS/"raylib-6.0_macos"
    subprocess.run(["clang", "-O2", "-std=c11", "-Wall", "-DPLATFORM_DESKTOP", f"-I{raylib/'include'}",
                    str(ROOT/"site/viewer/viewer.c"), str(raylib/"lib/libraylib.a"),
                    "-framework", "Cocoa", "-framework", "IOKit", "-framework", "CoreVideo",
                    "-framework", "OpenGL", "-lm", "-o", str(NATIVE/"viewer")], check=True)
    return NATIVE/"viewer"


def emsdk_env():
    env = os.environ.copy()
    emsdk = ROOT/".local/emsdk"
    out = subprocess.check_output(["bash", "-c", f"source {emsdk}/emsdk_env.sh >/dev/null 2>&1; env"], text=True)
    env.update(line.split("=", 1) for line in out.splitlines() if "=" in line)
    return env


def build_web(data_folder):
    out = SITE/"viewer"
    out.mkdir(parents=True, exist_ok=True)
    raylib = DEPS/"raylib-6.0_webassembly"
    subprocess.run(["emcc", str(ROOT/"site/viewer/viewer.c"), "-o", str(out/"viewer.js"), "-std=c11", "-O3",
                    f"-I{raylib/'include'}", str(raylib/"lib/libraylib.web.a"),
                    "-DPLATFORM_WEB", "-DGRAPHICS_API_OPENGL_ES3",
                    "-sUSE_GLFW=3", "-sUSE_WEBGL2=1", "-sMIN_WEBGL_VERSION=2", "-sMAX_WEBGL_VERSION=2",
                    "-sALLOW_MEMORY_GROWTH=1", "-sINITIAL_MEMORY=64MB", "-sSTACK_SIZE=1MB", "-sENVIRONMENT=web",
                    "-sEXPORTED_RUNTIME_METHODS=ccall,cwrap", "--preload-file", f"{data_folder}@data"],
                   check=True, env=emsdk_env())
    return out


def media(images, videos):
    """Convert repository figures (Git LFS PNG) to WebP and copy videos for the page."""
    from PIL import Image
    folder = SITE/"media"
    folder.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, source in images.items():
        image = Image.open(ROOT/source).convert("RGB")
        if image.width > 1800:
            image = image.resize((1800, round(image.height*1800/image.width)), Image.Resampling.LANCZOS)
        image.save(folder/f"{name}.webp", quality=84, method=6)
        out[name] = f"media/{name}.webp"
    for name, source in videos.items():
        shutil.copy2(ROOT/source, folder/f"{name}.mp4")
        out[name] = f"media/{name}.mp4"
    return out


IMAGES = {
    "workcell": "previews/neural_insertion/static_v1/overview.png",
    "mechanism": "previews/neural_insertion/static_v1/mechanism.png",
    "tool": "previews/neural_insertion/static_v1/tool_clearance.png",
    "cassette": "previews/neural_insertion/static_v1/thread_fixture.png",
    "task_regime": "previews/neural_insertion/task_regime_v1/quality_review.png",
    "approach_frame": "previews/neural_insertion/approach_v1/frame_0063.png",
    "learned_frame": "previews/neural_insertion/align_v3/seed1000009_01.13s.png",
    "learned_card": "previews/neural_insertion/align_v3/results_card.png",
}
VIDEOS = {
    "approach_tour": "previews/neural_insertion/approach_v1/approach_tour.mp4",
    "learned_alignment": "previews/neural_insertion/align_v3/learned_alignment.mp4",
}


GLOSSARY = ROOT/"previews/neural_insertion/glossary_v1"


def glossary(data):
    """Parts gallery: rendered views (Git LFS PNG) to WebP, captions from glossary.json."""
    from PIL import Image
    spec = json.loads((GLOSSARY/"glossary.json").read_text())
    folder = SITE/"media/glossary"
    folder.mkdir(parents=True, exist_ok=True)
    for part in spec["parts"]:
        Image.open(GLOSSARY/part["image"]).convert("RGB").save(folder/f"{part['id']}.webp", quality=84, method=6)
        part["media"] = f"media/glossary/{part['id']}.webp"
    (data/"glossary.json").write_text(json.dumps(spec)+"\n")


def assemble(media_paths):
    shutil.copy2(ROOT/"site/style.css", SITE/"style.css")
    viewer = hashlib.sha256(b"".join((SITE/"viewer"/n).read_bytes() for n in ("viewer.js", "viewer.wasm", "viewer.data")))
    (SITE/"journal.js").write_text((ROOT/"site/journal.js").read_text().replace("__VIEWER_BUILD__", viewer.hexdigest()[:12]))
    # Version the page's own script and stylesheet too, so a rebuild is never served from cache.
    page = (ROOT/"site/index.html").read_text()
    for name in ("journal.js", "style.css"):
        tag = hashlib.sha256((SITE/name).read_bytes()).hexdigest()[:12]
        page = page.replace(f'"{name}"', f'"{name}?v={tag}"')
    (SITE/"index.html").write_text(page)
    data = SITE/"data"
    data.mkdir(exist_ok=True)
    results = json.loads((ROOT/"docs/neural_insertion/ALIGN_RESULTS.json").read_text())
    clock = json.loads((ROOT/"docs/neural_insertion/THREAD_CLOCK_RESULTS.json").read_text())
    (data/"training.json").write_text(json.dumps({k: {"outcome": v["outcome"], "history": v["history"]}
                                                  for k, v in results["runs"].items()})+"\n")
    robust_path = ROOT/"docs/neural_insertion/ROBUST_ALIGN_RESULTS.json"
    if robust_path.exists():
        robust = json.loads(robust_path.read_text())
        runs = {"align_v3": {"history": results["runs"]["align_v3"]["history"]}}
        runs.update({k: {"history": v["history"]} for k, v in robust["runs"].items() if k >= "align_v4"})
        (data/"robust_training.json").write_text(json.dumps(runs)+"\n")
        (data/"robustness.json").write_text(json.dumps(robust["evaluation"])+"\n")
        (data/"compute.json").write_text(json.dumps({"compute": robust["compute"], "hardware": robust["hardware"],
                                                     "runs": {k: {"change": v["change"], "outcome": v["outcome"]}
                                                              for k, v in robust["runs"].items()}})+"\n")
    (data/"evaluation.json").write_text(json.dumps({k: v["summary"] for k, v in results["evaluation"].items()})+"\n")
    (data/"clock.json").write_text(json.dumps([{k: c[k] for k in ("material", "integrator", "time_constant_s", "dt_s",
                                                                  "difference_um", "passed")}
                                               for c in clock["comparisons"].values() if not c.get("units")])+"\n")
    glossary(data)
    (SITE/"media.json").write_text(json.dumps(media_paths, indent=1)+"\n")
    (SITE/".nojekyll").write_text("")


def manifest():
    files = {str(p.relative_to(SITE)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted(SITE.rglob("*")) if p.is_file() and p.name != "build.json"}
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain", "--", "site", "scripts/build_site.py",
                                          "src/sixlegs/neural_insertion/site_export.py"], cwd=ROOT))
    report = {"source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "source_dirty": dirty, "raylib": "6.0", "emscripten": "6.0.9", "files": files,
              "note": "Replays recorded MuJoCo states; the page does not simulate."}
    (SITE/"build.json").write_text(json.dumps(report, indent=1)+"\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-only", action="store_true")
    parser.add_argument("--robust", type=Path, help="robust policy weights (default: the committed align_v4 asset)")
    args = parser.parse_args()
    if args.robust:
        WEIGHTS["robust"] = args.robust
    data_folder = ROOT/"build/site_data"
    print(json.dumps(export_data(data_folder), indent=1))
    print("native viewer:", build_native())
    if args.native_only:
        return
    if SITE.exists():
        shutil.rmtree(SITE)
    SITE.mkdir(parents=True)
    print("web viewer:", build_web(data_folder))
    assemble(media(IMAGES, VIDEOS))
    report = manifest()
    print(f"site: {SITE} ({len(report['files'])} files)")


if __name__ == "__main__":
    main()
