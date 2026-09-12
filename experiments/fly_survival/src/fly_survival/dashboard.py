"""Local, dependency-free inspector. Video and telemetry share simulation time."""

import functools
import http.server
import json
import shutil
from pathlib import Path

from .paths import PREVIEWS


def export_static(n_flies=8):
    root = PREVIEWS / "inspector"
    root.mkdir(parents=True, exist_ok=True)
    for source, name in [
        ("scene_overview_v1.png", "arena.png"),
        ("brain_atlas_v1.png", "brain.png"),
    ]:
        shutil.copy2(PREVIEWS / source, root / name)
    shutil.copy2(Path(__file__).with_name("inspector.html"), root / "index.html")
    (root / "telemetry.json").write_text(
        json.dumps(
            {
                "mode": "static",
                "n_flies": n_flies,
                "frames": [],
                "neural": {
                    "neurons": 166700,
                    "connections": 25582938,
                    "located": 140638,
                },
            }
        )
    )
    return root


def serve(port=8794):
    root = PREVIEWS / "inspector"
    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler, directory=str(root)
    )
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    print(f"Fly inspector: http://127.0.0.1:{port}", flush=True)
    server.serve_forever()
