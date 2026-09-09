"""Vendor the unarmed Go2 reference model at the repository's pinned revision."""

import concurrent.futures
import hashlib
import json

from fetch_assets import REPO, REV, ROOT, fetch


def main():
    tree = json.loads(
        fetch(f"https://api.github.com/repos/{REPO}/git/trees/{REV}?recursive=1")
    )
    paths = [
        e["path"]
        for e in tree["tree"]
        if e["type"] == "blob"
        and e["path"].startswith("unitree_go2/")
        and (
            "/assets/" in e["path"]
            or e["path"].endswith((".xml", "LICENSE", "README.md"))
        )
    ]

    def download(path):
        target = ROOT / "assets/menagerie" / path
        data = fetch(f"https://raw.githubusercontent.com/{REPO}/{REV}/{path}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return path, hashlib.sha256(data).hexdigest()

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        hashes = dict(pool.map(download, paths))
    (ROOT / "assets/menagerie/unitree_go2_manifest.json").write_text(
        json.dumps(
            {
                "repository": f"https://github.com/{REPO}",
                "revision": REV,
                "sha256": hashes,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Vendored {len(paths)} Go2 files at {REV}")


if __name__ == "__main__":
    main()
