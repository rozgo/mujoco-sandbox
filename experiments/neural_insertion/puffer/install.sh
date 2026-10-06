#!/usr/bin/env bash
# Install the surgical alignment environment into a PufferLib checkout at the
# pinned revision and build it. Usage: install.sh PUFFER_DIR [build.sh flags]
#   (no flags)  native CUDA trainer -> PUFFER_DIR/puffer   (Linux + NVIDIA)
#   --cpu       CPU evaluation binary                      (any platform)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
PUFFER=$(cd "$1" && pwd)
shift
PIN=6ffa5b10dbbbe4d1e8288367c7d9d3acd3bad4a2
if [ "$(git -C "$PUFFER" rev-parse HEAD)" != "$PIN" ]; then
    echo "PufferLib checkout must be at $PIN" >&2
    exit 1
fi
MJ=$(cd "$REPO" && uv run --locked python -c 'import mujoco, pathlib; print(pathlib.Path(mujoco.__file__).parent)')
LIB=$(find "$MJ" -maxdepth 1 \( -name 'libmujoco.so.*' -o -name 'libmujoco.*.dylib' \) | head -1)
XML=$(cd "$REPO" && uv run --locked python -c 'from sixlegs.neural_insertion.rl_scene import build_rl_scene; print(build_rl_scene()[0])')
mkdir -p "$PUFFER/ocean/surgical_align" "$PUFFER/resources/surgical_align"
cp "$HERE/surgical_align/surgical_align.h" "$REPO/src/sixlegs/neural_insertion/native/surgical_core.h" \
    "$PUFFER/ocean/surgical_align/"
cp "$HERE/surgical_align.ini" "$PUFFER/config/surgical_align.ini"
cp "$XML" "$PUFFER/resources/surgical_align/surgical_align.xml"
LINKDIR="$REPO/build/neural_insertion/mujoco_link"
mkdir -p "$LINKDIR"
if [ "$(uname -s)" = "Darwin" ]; then
    # The wheel's dylib is named @rpath/mujoco.framework/Versions/A/<lib>; give a
    # standalone binary that layout through symlinks in ignored build storage.
    mkdir -p "$LINKDIR/mujoco.framework/Versions/A"
    ln -sf "$LIB" "$LINKDIR/mujoco.framework/Versions/A/$(basename "$LIB")"
    LINK="$LIB -Xlinker -rpath -Xlinker $LINKDIR"
else
    # nvcc rejects a versioned .so as an input; link through an unversioned name.
    ln -sf "$LIB" "$LINKDIR/libmujoco.so"
    LINK="-L$LINKDIR -lmujoco -Xlinker -rpath -Xlinker $MJ"
fi
# build.sh forwards NVCC_EXTRA to both the CUDA trainer and the CPU build.
export NVCC_EXTRA="-I$MJ/include $LINK"
cd "$PUFFER"
# build.sh's only bash-4 construct is ${ENV^^}. Under macOS bash 3.2, run an
# untracked local copy with that one expansion rewritten; the pinned file is untouched.
BUILD=./build.sh
if [ "${BASH_VERSINFO[0]}" -lt 4 ]; then
    sed 's/-DPUFFER_\${ENV^^}/-DPUFFER_$(echo "$ENV" | tr a-z A-Z)/' build.sh > .build_bash3.sh
    BUILD=./.build_bash3.sh
fi
bash "$BUILD" surgical_align "$@"
sha256sum ocean/surgical_align/*.h config/surgical_align.ini resources/surgical_align/surgical_align.xml 2>/dev/null \
    || shasum -a 256 ocean/surgical_align/*.h config/surgical_align.ini resources/surgical_align/surgical_align.xml
