#!/usr/bin/env bash
# Link a source-frozen diagnostic against a verified immutable engine build.
set -euo pipefail
repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$repo/scripts/ds4-research-root.sh"
ds4_resolve_research_roots "$repo"
engine_rev=${1:?usage: run-glm53-q4-pair-probe.sh ENGINE_COMMIT7 REP}
repeat=${2:?missing repetition}
[[ $engine_rev =~ ^[0-9a-f]{7}$ && $repeat =~ ^[1-9][0-9]*$ ]]
revision=$(git -C "$repo" rev-parse HEAD)
engine=$DS4_RESEARCH_ROOT/builds/glm53-full-$engine_rev
[[ $(cut -c1-7 "$engine/BUILD-SOURCE") == "$engine_rev" ]]
(cd "$engine" && sha256sum --check --status BUILD-SHA256SUMS)
git -C "$repo" diff --quiet
git -C "$repo" diff --cached --quiet
artifact=$DS4_RESEARCH_ROOT/builds/glm53-q4-pair-${revision:0:7}-$engine_rev-r$repeat
[[ ! -e $artifact ]]
mkdir -p "$artifact"
git -C "$repo" archive HEAD tests/test_rocm_glm5_q4k_shard_compose.cu \
    tests/glm5_gguf_test.hpp tests/glm5_q4k_pair_probe.hpp ds4_glm5_kda.h \
    ds4_glm5_next_runtime.h ds4_tp.h ds4.h ds4_ssd.h ds4_gpu.h ds4_gpu_mgpu.h \
    ds4_gpu_args.h scripts/run-glm53-q4-pair-probe.sh | tar -x -C "$artifact"
cd "$artifact"
git -C "$repo" rev-parse HEAD > TEST-SOURCE
cp "$engine/BUILD-SOURCE" ENGINE-SOURCE
hip=${DS4_ROCM_HOME:-/home/wkljohn/Desktop/cc/toolchains/rocm-10.0.0-gfx1151/install}
"$hip/bin/hipcc" --version > compiler.txt
if ! cmp -s compiler.txt "$engine/compiler.txt"; then
    echo "error: probe toolchain differs from frozen engine; set DS4_ROCM_HOME to its compiler" >&2
    exit 1
fi
compile=(-O3 -fno-fast-math -ffp-contract=off --offload-arch=gfx1151
         -mno-wavefrontsize64 -DDS4_ROCM_BUILD -I. -c tests/test_rocm_glm5_q4k_shard_compose.cu -o test.o)
objects=("$engine/ds4_rocm.o" "$engine/ds4_rocm_compat.o"
         "$engine/ds4_rocm_unavailable.o" "$engine/ds4_glm5_next_runtime.o"
         "$engine/tests/ds4_tp_hello_test.o")
link=(-O3 --offload-arch=gfx1151 -mno-wavefrontsize64 test.o "${objects[@]}"
      -Wl,--gc-sections "-L$hip/lib" "-Wl,-rpath,$hip/lib"
      -lm -pthread -lhipblas -lhipblaslt -o probe)
printf '%q ' "$hip/bin/hipcc" "${compile[@]}" > COMPILE-COMMAND
printf '%q ' "$hip/bin/hipcc" "${link[@]}" > LINK-COMMAND
"$hip/bin/hipcc" "${compile[@]}" > build.log 2>&1
"$hip/bin/hipcc" "${link[@]}" >> build.log 2>&1
sha256sum probe test.o "${objects[@]}" TEST-SOURCE ENGINE-SOURCE \
    COMPILE-COMMAND LINK-COMMAND compiler.txt tests/* *.h scripts/* > BUILD-SHA256SUMS
export DS4_GLM5_MODEL=/home/wkljohn/Desktop/cc/models/antirez-glm-5.3-flash-gguf/GLM-5.3-Flash-Q4_K.gguf
[[ $(stat -c '%s' "$DS4_GLM5_MODEL") == 190875526464 ]]
cat > MODEL-IDENTITY <<EOF
path=$DS4_GLM5_MODEL
size=$(stat -c '%s' "$DS4_GLM5_MODEL")
sample_fingerprint=51f8162bbc0399c1fad7bb63e65244f6151aaea392bc5c7fb469650762283621
EOF
for mode in 0 1 2; do
    DS4_GLM5_Q4K_PAIR_PROBE=$mode DS4_GLM5_Q4K_PAIR_OUTPUT=$artifact/mode$mode.bin \
        ./probe > mode$mode.log 2>&1
    cat mode$mode.log
    if (( mode > 0 )); then
        cmp mode0.bin mode$mode.bin
        printf 'PASS complete mid/down/output equality mode0 vs mode%s\n' "$mode"
    fi
done
sha256sum mode*.bin mode*.log > RESULT-SHA256SUMS
printf 'artifact=%s\n' "$artifact"
