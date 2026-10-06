#!/usr/bin/env bash
# Build the checkpoint evaluator against a pinned PufferLib checkout's CPU
# network code and the shared alignment core. Usage: build_eval.sh PUFFER_DIR OUT_BINARY
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
PUFFER=$(cd "$1" && pwd)
OUT=$2
PIN=6ffa5b10dbbbe4d1e8288367c7d9d3acd3bad4a2
[ "$(git -C "$PUFFER" rev-parse HEAD)" = "$PIN" ] || { echo "PufferLib checkout must be at $PIN" >&2; exit 1; }
MJ=$(cd "$REPO" && uv run --locked python -c 'import mujoco, pathlib; print(pathlib.Path(mujoco.__file__).parent)')
LIB=$(find "$MJ" -maxdepth 1 \( -name 'libmujoco.so.*' -o -name 'libmujoco.*.dylib' \) | head -1)
LINKDIR="$REPO/build/neural_insertion/mujoco_link"
mkdir -p "$LINKDIR"
if [ "$(uname -s)" = "Darwin" ]; then
    mkdir -p "$LINKDIR/mujoco.framework/Versions/A"
    ln -sf "$LIB" "$LINKDIR/mujoco.framework/Versions/A/$(basename "$LIB")"
    LINK=("$LIB" -Wl,-rpath,"$LINKDIR")
else
    LINK=("$LIB" -Wl,-rpath,"$MJ")
fi
# puffercpu.c's net code needs the raylib headers that build.sh downloads.
RAYLIB=$(find "$PUFFER" -maxdepth 1 -type d -name 'raylib-5.5_*' | head -1)
${CC:-clang} -O2 -std=gnu11 -w -I"$PUFFER/src" -I"$PUFFER/vendor" -I"$RAYLIB/include" \
    -I"$REPO/src/sixlegs/neural_insertion/native" -I"$MJ/include" \
    "$HERE/align_eval.c" "${LINK[@]}" -lm -lpthread -o "$OUT"
echo "Built: $OUT"
