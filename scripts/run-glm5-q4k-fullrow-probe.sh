#!/usr/bin/env bash
# Test-only research; no production dispatch or public performance claim.
set -euo pipefail
probe_repo=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "$probe_repo/scripts/ds4-research-root.sh"
ds4_resolve_research_roots "$probe_repo"
probe_revision=${1:?explicit clean source commit required}
probe_tag=${2:?unique artifact tag required}
[[ $probe_revision =~ ^[0-9a-f]{7,40}$ && $probe_tag =~ ^[a-zA-Z0-9._-]+$ ]]
cd "$probe_repo"
[[ $(git rev-parse HEAD) == "$(git rev-parse "$probe_revision")" ]]
[[ -z $(git status --porcelain) ]]
: "${DS4_GLM5_MODEL:?unchanged GGUF path required}"
probe_parent=$DS4_RESEARCH_ROOT/candidates/glm53-uncensored-20260915/research
[[ -d $probe_parent ]]
probe_dir=$probe_parent/$probe_tag
[[ ! -e $probe_dir ]]
if pgrep -x 'ds4|ds4-bench-tp|ds4-server|score_official|rocprofv3' >/dev/null; then
    echo 'Inference/profiling is active; refusing concurrent microbenchmark.' >&2
    exit 1
fi
if ssh -o BatchMode=yes -o ConnectTimeout=10 "${DS4_PROBE_PEER:-wkljohn@192.168.99.2}" \
    'pgrep -x "ds4|ds4-bench-tp|ds4-server|score_official|rocprofv3" >/dev/null'; then
    echo 'Peer inference/profiling is active; refusing concurrent microbenchmark.' >&2
    exit 1
else
    probe_status=$?
    [[ $probe_status == 1 ]]
fi
mkdir "$probe_dir"
probe_hip=${HIPCC:-/opt/rocm/bin/hipcc}
probe_flags=(-O3 -pthread -D__HIP_PLATFORM_AMD__ --offload-arch=gfx1151
    -mno-wavefrontsize64 -DDS4_GFX1151_WAVE32=1 -I.)
git rev-parse HEAD > "$probe_dir/source.txt"
"$probe_hip" --version > "$probe_dir/compiler.txt"
stat -Lc 'model=%n bytes=%s mtime=%y inode=%i' "$DS4_GLM5_MODEL" > "$probe_dir/model-stat.txt"
sha256sum tests/glm5_q4k_fullrow_kernels.cu tests/test_glm5_q4k_fullrow.cu \
    tests/glm5_q4k_fullrow_test.hpp tests/glm5_gguf_test.hpp \
    rocm/ds4_rocm_q4k_dot.cuh rocm/ds4_rocm_q4k_types.cuh ds4_rocm.h \
    scripts/run-glm5-q4k-fullrow-probe.sh > "$probe_dir/source-SHA256SUMS"
exec 3> "$probe_dir/commands.log"
export BASH_XTRACEFD=3
set -x
"$probe_hip" "${probe_flags[@]}" -ffast-math -fno-finite-math-only \
    -c tests/glm5_q4k_fullrow_kernels.cu -o "$probe_dir/kernels.o" \
    > "$probe_dir/kernel-build.log" 2>&1
"$probe_hip" "${probe_flags[@]}" -fno-fast-math -ffp-contract=off \
    -c tests/test_glm5_q4k_fullrow.cu -o "$probe_dir/host.o" \
    > "$probe_dir/host-build.log" 2>&1
"$probe_hip" --offload-arch=gfx1151 "$probe_dir/host.o" "$probe_dir/kernels.o" \
    -o "$probe_dir/test" > "$probe_dir/link.log" 2>&1
sha256sum "$probe_dir/test" "$probe_dir/host.o" "$probe_dir/kernels.o" \
    > "$probe_dir/SHA256SUMS"
set +e
"$probe_dir/test" > "$probe_dir/test.log" 2>&1
probe_exit=$?
set -e
printf 'exit_code=%s\n' "$probe_exit" > "$probe_dir/status.txt"
set +x
echo "exit_code=$probe_exit evidence=$probe_dir"
exit "$probe_exit"
