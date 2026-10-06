"""Build the journal (build/site): page, styles, script, media and the baseline data.

Media come from previews/ (Git LFS): stills converted to WebP, videos copied. The baseline table reads
docs/neural_insertion/TUBE_BASELINE.json. The previous journal (first-pass mechanism and alignment training,
with the in-browser replay viewer) is in Git history up to commit 966a2ff; site/viewer/ is not part of this
journal.

Usage: uv run --locked python scripts/build_site.py
"""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT/"build/site"
PREVIEWS = ROOT/"previews/neural_insertion/tube_design"
STILLS = ("1_tool", "2_waiting", "3_inserted", "4_three_placed")
VIDEOS = ("three_sites_level0", "three_sites_level1", "three_sites_level2")


def main():
    if SITE.exists():
        shutil.rmtree(SITE)
    (SITE/"media").mkdir(parents=True)
    (SITE/"data").mkdir()
    for name in STILLS:
        Image.open(PREVIEWS/f"stills/{name}.png").convert("RGB").save(SITE/f"media/{name}.webp", quality=88)
    for name in VIDEOS:
        shutil.copy2(PREVIEWS/f"{name}.mp4", SITE/f"media/{name}.mp4")
    baseline = json.loads((ROOT/"docs/neural_insertion/TUBE_BASELINE.json").read_text())
    (SITE/"data/baseline.json").write_text(json.dumps({"summary": baseline["summary"], "runs": baseline["runs"]})+"\n")
    sources = [ROOT/"site"/n for n in ("index.html", "style.css", "journal.js")]
    build = hashlib.sha256(b"".join(p.read_bytes() for p in sources)
                           + (SITE/"data/baseline.json").read_bytes()).hexdigest()[:12]
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    for p in sources:
        text = p.read_text().replace("__BUILD__", build).replace("__COMMIT__", commit)
        (SITE/p.name).write_text(text)
    size = sum(f.stat().st_size for f in SITE.rglob("*") if f.is_file())
    print(f"built {SITE} ({size/1e6:.1f} MB, build {build}, commit {commit})")


if __name__ == "__main__":
    main()
