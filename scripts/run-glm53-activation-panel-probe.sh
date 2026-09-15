#!/usr/bin/env bash
# Immutable local-GPU diagnostic. Reserve both TP nodes before invoking.
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$repo/scripts/ds4-research-root.sh"
ds4_resolve_research_roots "$repo"
revision=$(git -C "$repo" rev-parse HEAD)
repeat=${1:?usage: run-glm53-activation-panel-probe.sh REP}
math=${2:-strict}
[[ $repeat =~ ^[1-9][0-9]*$ ]]
[[ $math == strict || $math == production ]]
git -C "$repo" diff --quiet
git -C "$repo" diff --cached --quiet
artifact=$DS4_RESEARCH_ROOT/builds/glm53-activation-panel-${revision:0:7}-r$repeat
if [[ $math == production ]]; then artifact+=-production; fi
hip=${DS4_ROCM_HOME:-/home/wkljohn/Desktop/cc/toolchains/rocm-7.14.0-gfx1151/install}
[[ ! -e $artifact && -x $hip/bin/hipcc ]]
mkdir -p "$artifact"
git -C "$repo" archive "$revision" \
    rocm/ds4_rocm_bf16_toktile.cuh tests/glm5_gguf_test.hpp \
    scripts/glm5_bf16_activation_panel_bench.cu \
    scripts/run-glm53-activation-panel-probe.sh | tar -x -C "$artifact"
cd "$artifact"
git -C "$repo" rev-parse HEAD > BUILD-SOURCE
"$hip/bin/hipcc" --version > compiler.txt
args=(-O3 -fno-fast-math -ffp-contract=off --offload-arch=gfx1151
      -mno-wavefrontsize64 -I. -MD -MF dependencies.d)
if [[ $math == production ]]; then
    args+=(-ffast-math -fno-finite-math-only -DDS4_GFX1151_WAVE32=1)
fi
args+=(scripts/glm5_bf16_activation_panel_bench.cu -o probe)
printf '%q ' "$hip/bin/hipcc" "${args[@]}" > BUILD-COMMAND
printf '\n' >> BUILD-COMMAND
"$hip/bin/hipcc" "${args[@]}" > build.log 2>&1
sha256sum probe BUILD-SOURCE BUILD-COMMAND compiler.txt dependencies.d \
    rocm/ds4_rocm_bf16_toktile.cuh tests/glm5_gguf_test.hpp scripts/* \
    > BUILD-SHA256SUMS
export DS4_GLM5_MODEL=/home/wkljohn/Desktop/cc/models/antirez-glm-5.3-flash-gguf/GLM-5.3-Flash-Q4_K.gguf
stat --printf='model=%n\nsize=%s\nmtime=%y\n' "$DS4_GLM5_MODEL" > MODEL-IDENTITY
./probe > probe.log 2>&1
sha256sum probe.log MODEL-IDENTITY > RESULT-SHA256SUMS
printf 'artifact=%s\n' "$artifact"
cat probe.log
