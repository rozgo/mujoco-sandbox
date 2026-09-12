"""Download auditable official inputs and bake the measured anatomical positions."""

import base64
import hashlib
import json
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image, ImageDraw

from .paths import NEURAL_DATA, PREVIEWS, VENDOR

BUCKET = (
    "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome"
)
FILES = [
    "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "body-neurotransmitters-male-cns-v1.0.feather",
    "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
]


def download(name, url):
    target = NEURAL_DATA / "raw" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    manifest = target.with_suffix(target.suffix + ".source.json")
    if target.exists() and manifest.exists():
        report = json.loads(manifest.read_text())
        if (
            hashlib.file_digest(target.open("rb"), "sha256").hexdigest()
            == report["sha256"]
        ):
            return report
    start = time.perf_counter()
    req = urllib.request.Request(url, headers={"User-Agent": "FlySurvivalResearch/0.1"})
    with urllib.request.urlopen(req, timeout=90) as response:
        expected = int(response.headers.get("Content-Length", 0))
        provider_hash = response.headers.get("x-goog-hash", "")
        sha, md5 = hashlib.sha256(), hashlib.md5()
        partial = target.with_suffix(target.suffix + ".part")
        size = 0
        with partial.open("wb") as out:
            while block := response.read(4 * 1024 * 1024):
                out.write(block)
                sha.update(block)
                md5.update(block)
                size += len(block)
        if expected and size != expected:
            raise RuntimeError(f"Incomplete download: {name}")
        expected_md5 = next(
            (
                s.strip()[4:]
                for s in provider_hash.split(",")
                if s.strip().startswith("md5=")
            ),
            None,
        )
        if expected_md5 and base64.b64encode(md5.digest()).decode() != expected_md5:
            raise RuntimeError(f"Provider checksum mismatch: {name}")
        partial.replace(target)
    report = {
        "name": name,
        "url": url,
        "bytes": size,
        "sha256": sha.hexdigest(),
        "provider_md5_verified": bool(expected_md5),
        "seconds": time.perf_counter() - start,
    }
    manifest.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    if name.startswith("body-annotations"):
        bake_anatomy(target)
    return report


def bake_anatomy(path):
    from pyarrow import feather

    ann = feather.read_table(path).to_pandas()
    ann = (
        ann.loc[ann.superclass.notna() & ann.superclass.ne("")]
        .drop_duplicates("bodyId")
        .sort_values("bodyId")
    )
    positions, indices = [], []
    for i, row in enumerate(ann.itertuples()):
        loc = row.somaLocation
        if not isinstance(loc, (list, np.ndarray)) or len(loc) != 3:
            loc = row.tosomaLocation
        if (
            isinstance(loc, (list, np.ndarray))
            and len(loc) == 3
            and np.isfinite(loc).all()
        ):
            positions.append([loc[0], loc[2]])
            indices.append(i)
    xy = np.asarray(positions, dtype=float)
    lo, hi = np.nanpercentile(xy, [0.2, 99.8], axis=0)
    norm = np.clip(
        (xy - lo + ((hi - lo).max() - (hi - lo)) / 2) / (hi - lo).max(), 0, 1
    )
    image = Image.new("RGB", (900, 900), "#141616")
    draw = ImageDraw.Draw(image)
    bins = np.zeros((840, 840), dtype=float)
    pixel = np.minimum((norm * 839).astype(int), 839)
    np.add.at(bins, (pixel[:, 1], pixel[:, 0]), 1)
    density = np.log1p(bins) / max(np.log1p(bins).max(), 1)
    base = np.asarray(image).copy()
    base[30:870, 30:870] += (density[..., None] * np.array([190, 181, 153])).astype(
        np.uint8
    )
    image = Image.fromarray(base)
    draw = ImageDraw.Draw(image)
    draw.text((24, 18), "MaleCNS v1.0 / anatomical positions", fill="#b9b6ad")
    draw.text(
        (24, 870),
        f"{len(indices):,} located / {len(ann):,} neurons / activity overlay pending",
        fill="#b9b6ad",
    )
    PREVIEWS.mkdir(parents=True, exist_ok=True)
    image.save(PREVIEWS / "brain_atlas_v1.png")
    np.savez_compressed(
        NEURAL_DATA / "atlas.npz",
        indices=indices,
        xy=norm.astype(np.float32),
        n_neurons=len(ann),
    )
    print(f"Baked measured anatomy: {len(indices):,} located neurons", flush=True)


def prepare(build=True):
    start = time.perf_counter()
    sources = [(name, f"{BUCKET}/{name}") for name in FILES]
    sources.append(
        (
            "optic-columns.xlsx",
            "https://raw.githubusercontent.com/flyconnectome/2025malecns/67767d2233657983993ff6c2be48e836a935863c/supplemental_data/optic-column-type-assignments-v1.0.xlsx",
        )
    )
    with ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(lambda s: download(*s), sources))
    if build:
        os.environ["FLY_DATA"] = str(NEURAL_DATA)
        sys.path.insert(0, str(VENDOR))
        import build_brain

        build_brain.main()
    (NEURAL_DATA / "provenance.json").write_text(
        json.dumps(
            {
                "inputs": records,
                "seconds": time.perf_counter() - start,
                "built": build,
                "connectome": "MaleCNS v1.0",
                "license": "CC BY 4.0",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    prepare("--download-only" not in sys.argv)
