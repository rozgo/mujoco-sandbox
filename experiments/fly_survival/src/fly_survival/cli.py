import argparse
import json
import time

from PIL import Image

from .paths import PREVIEWS


def main():
    parser = argparse.ArgumentParser(description="Fly survival lab")
    parser.add_argument("command", choices=["preview", "prepare", "inspect"])
    parser.add_argument("--flies", type=int, default=8)
    args = parser.parse_args()
    if args.command == "inspect":
        from .dashboard import serve

        serve()
        return
    if args.command == "prepare":
        from .prepare import prepare

        prepare()
        return
    from .scene import build, render

    start = time.perf_counter()
    arena = build(args.flies)
    PREVIEWS.mkdir(parents=True, exist_ok=True)
    for view in ("overview", "swatter", "refuge"):
        Image.fromarray(render(arena, view=view)).save(
            PREVIEWS / f"scene_{view}_v1.png"
        )
    model = arena.sim.mj_model
    report = {
        "n_flies": args.flies,
        "nq": model.nq,
        "nv": model.nv,
        "nu": model.nu,
        "nbody": model.nbody,
        "physics_steps": 0,
        "seconds": time.perf_counter() - start,
        "units": "millimeter, gram, second",
        "fly_mass_kg": float(model.body_subtreemass[arena.fly_body_ids[0]]) * 0.001,
    }
    (PREVIEWS / "scene_v1.json").write_text(json.dumps(report, indent=2) + "\n")
    from .dashboard import export_static

    export_static(args.flies)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
