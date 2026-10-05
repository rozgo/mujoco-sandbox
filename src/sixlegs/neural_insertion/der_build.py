"""Reproducible optional DER plugin build; no installed MuJoCo changes."""

import hashlib
import io
import json
from pathlib import Path
import platform
import subprocess
import tarfile
import urllib.request

import mujoco

from .scene import ROOT

EIGEN_REV = "464c1d097891a1462ab28bf8bb763c1683883892"
EIGEN_SHA = "d9a483c02dbe2bee0c003aab550bfa8670f848e3eb8e9c36a9e3c30315bf77df"
SOURCE = Path(__file__).parent
VENDOR = SOURCE / "der_vendor"


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f"Upstream patch context changed: {old[:60]}")
    return text.replace(old, new, 1)


def build():
    if mujoco.__version__ != "3.12.0":
        raise RuntimeError("This port is validated only against MuJoCo 3.12.0")
    upstream = json.loads((VENDOR / "UPSTREAM.json").read_text())
    for name, record in upstream["files"].items():
        if hashlib.sha256((VENDOR/name).read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Changed upstream source: {name}")
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in [SOURCE/"der_build.py", SOURCE/"der_bridge.cc", *sorted(VENDOR.iterdir())]
              if p.is_file()}
    key = hashlib.sha256(json.dumps([hashes, mujoco.__version__, platform.machine(),
                                    platform.system(), EIGEN_SHA], sort_keys=True).encode()).hexdigest()[:16]
    folder = ROOT / "build/neural_insertion/der_plugin" / key
    library = folder / ("libder.dylib" if platform.system() == "Darwin" else "libder.so")
    if library.exists() and (folder/"build.json").exists():
        return library
    deps = ROOT / "build/neural_insertion/der_deps"
    deps.mkdir(parents=True, exist_ok=True)
    archive = deps / "eigen.tar.gz"
    if not archive.exists():
        url = f"https://gitlab.com/libeigen/eigen/-/archive/{EIGEN_REV}/eigen-{EIGEN_REV}.tar.gz"
        with urllib.request.urlopen(url, timeout=60) as response:
            archive.write_bytes(response.read())
    blob = archive.read_bytes()
    if hashlib.sha256(blob).hexdigest() != EIGEN_SHA:
        raise ValueError("Eigen download hash mismatch")
    eigen = deps / ("eigen-" + EIGEN_REV)
    # Re-extract the checked archive so existing scratch headers are not trusted.
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tar:
        tar.extractall(deps, filter="data")
    folder.mkdir(parents=True, exist_ok=True)
    for name in upstream["files"]:
        text = (VENDOR/name).read_text()
        text = text.replace("<mujoco/mjtnum.h>", "<mujoco/mjtype.h>")
        if name == "wire_qst.h":
            text = replace_once(text, "  WireQST(WireQST&&) = default;",
                                "  WireQST(WireQST&&) = default;\n  WireQST(const WireQST&) = default;")
        if name == "wire_qst.cc":
            text = replace_once(text, """  double dot_norm_val = v1.dot(v2) / (v1.norm() * v2.norm());
  if (dot_norm_val > 1.0) dot_norm_val = 1.0;
  double theta_diff = std::acos(dot_norm_val);
  if ((v1.cross(v2)).dot(va) < 0) {
    theta_diff *= -1.0;
  }""", """  double theta_diff = std::atan2(v1.cross(v2).dot(va.normalized()), v1.dot(v2));""")
            # Upstream reads Eigen vectors and theta before initializing them.
            text = replace_once(text, "  // Initialize twist displacement", """  for (auto& node : nodes) {
    node.kb.setZero(); node.phi_i = 0; node.k = 0;
  }
  for (auto& edge : edges) edge.theta = 0;

  // Initialize twist displacement""")
            # The material frame must use curvature at this evaluation's qpos,
            # rather than the preceding Compute call's curvature.
            text = replace_once(text, "  bigL_bar /= 2.;", """  bigL_bar /= 2.;
  for (int i=1; i<=nv; ++i) {
    nodes[i].phi_i = WireUtils::calculateAngleBetween(edges[i-1].e, edges[i].e);
    nodes[i].kb = 2*edges[i-1].e.cross(edges[i].e) /
      (edges[i-1].e_bar*edges[i].e_bar + edges[i-1].e.dot(edges[i].e));
  }""")
        (folder/name).write_text(text)
    mjdir = Path(mujoco.__file__).parent
    libraries = list(mjdir.glob("libmujoco.*.dylib" if platform.system() == "Darwin" else "libmujoco.so.*"))
    if len(libraries) != 1:
        raise RuntimeError("Expected one installed MuJoCo library")
    command = ["clang++" if platform.system() == "Darwin" else "c++", "-O2", "-std=c++17",
               "-shared", "-fPIC", "-DEIGEN_MPL2_ONLY", "-DEIGEN_INITIALIZE_MATRICES_BY_NAN",
               "-I"+str(mjdir/"include"), "-I"+str(eigen), "-I"+str(folder),
               str(SOURCE/"der_bridge.cc"), str(folder/"wire_qst.cc"), str(folder/"wire_utils.cc"),
               str(libraries[0]), "-Wl,-rpath,"+str(mjdir), "-o", str(library)]
    subprocess.run(command, check=True)
    (folder/"build.json").write_text(json.dumps({"source_sha256": hashes, "upstream": upstream,
        "mujoco": mujoco.__version__, "eigen_revision": EIGEN_REV, "eigen_archive_sha256": EIGEN_SHA,
        "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "compiler": subprocess.check_output([command[0], "--version"], text=True).splitlines()[0],
        "patches": ["3.12 numeric header rename", "initialize curvature/theta", "current-state curvature before frame transport",
                    "atan2 signed twist angle", "copy constructor", "public plugin lifecycle and state adapter"],
        "upstream_source_force_formula_changes": False,
        "projection_variants": {"published": "upstream lever torques",
                                "direct": "Cartesian Jacobian projection plus end twist moments"}}, indent=2)+"\n")
    return library


if __name__ == "__main__":
    print(build())
