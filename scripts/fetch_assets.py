"""Vendor only the two required Menagerie models, pinned to an immutable commit."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REV = "8161bba264d7fa7c99ca301e91e7fb44737676ad"
REPO = "google-deepmind/mujoco_menagerie"
MODELS = ("kinova_gen3", "robotiq_2f85")

def fetch(url):
    with urllib.request.urlopen(url, timeout=90) as response:
        return response.read()

def main():
    tree = json.loads(fetch(f"https://api.github.com/repos/{REPO}/git/trees/{REV}?recursive=1"))
    paths = [e["path"] for e in tree["tree"] if e["type"] == "blob"
             and e["path"].split("/")[0] in MODELS
             and ("/assets/" in e["path"] or e["path"].endswith((".xml", "LICENSE", "README.md")))]
    def download(path):
        target = ROOT / "assets/menagerie" / path
        data = fetch(f"https://raw.githubusercontent.com/{REPO}/{REV}/{path}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return path, hashlib.sha256(data).hexdigest()
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        hashes = dict(pool.map(download, paths))
    (ROOT / "assets/menagerie/manifest.json").write_text(json.dumps(
        {"repository": f"https://github.com/{REPO}", "revision": REV, "sha256": hashes}, indent=2) + "\n")
    print(f"Vendored {len(paths)} files at {REV}")

if __name__ == "__main__":
    main()
