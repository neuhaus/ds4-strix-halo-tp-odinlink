# Tracked DS4 experiments

This workflow covers ordinary inference, transport, kernel, and harness
research. DSpark-specific checks apply only when a DSpark candidate is in
scope. A local checkpoint records work; it is not permission to merge or push.

The original GLM5.3 Flash track requires Antirez's unchanged GGUF (2026-09-13).
The user subsequently selected `GLM-5.3-Flash-Uncensored-Q4_K-ds4.gguf` for a
distinct support and performance scope (2026-09-15). Preserve each artifact's
bytes and quantization. Establish a same-model baseline and quality reference;
the original artifact's promotion does not approve the selected artifact.
Do not substitute a compact/requantized model or requantize weights at runtime.
Compact-model tracks remain paused and unmerged. Mandatory RoCE v2, zero
payload fallback and no persistent expanded-weight cache still apply.

## Before a change

1. Record the named branch, parent commit, exact hypothesis, baseline run tags,
   intended variables, candidate lane, and smallest falsifiable test in a
   dossier under `$DS4_RESEARCH_ROOT`.
2. Preserve dirty work. Attribute changes with scoped commits or dated source
   snapshots; do not infer causality from a large HEAD-to-dirty diff.
3. Check both nodes for existing GPU jobs before reserving a test. Read the
   coordinator's first failure before diagnosing a worker connection refusal.

## Research-track lifecycle

Named research branches are durable tracks. Mark a branch `paused` or
`given-up` when its direction is stopped, but retain its source, evidence, and
status notes. Such a branch stays out of `main` and is not a release baseline.

Periodic audits may identify a useful idea in an older track. Transfer it by
starting a fresh branch from current `main` and recording an adaptation ledger
entry with the donor branch/commit, patch or source hash, decision, and new
commit. Re-test the adapted code under the current gates; never promote by
directly merging a paused or given-up branch.

## Source and build checkpoints

- Commit each scoped source or harness change before publishable timing. For an
  early diagnostic, preserve the complete source snapshot, tracked diff,
  untracked source files, parent, and SHA-256 inventory; label it diagnostic.
- Build into a unique artifact directory. Record the exact command,
  compiler/ROCm identity, defines for all compilation units, build log,
  dependency/helper inventory, and hashes of `ds4` and `ds4-bench-tp`.
- Freeze each artifact after building. Record both nodes' hashes, model and
  drafter fingerprints where applicable, provider, and feature negotiation.
  Rebuilding in place invalidates the old artifact identity.

## Matched tests

1. Run the smallest diagnostic that exercises the change. Pass engine controls
   as trailing `NAME=VALUE` launcher arguments and inspect both rank
   environments in the manifest.
2. Freeze a comparison recipe before examining performance: keep workload,
   prompt bytes, model, toolchain, provider, residency, split, and diagnostic
   state fixed. List the one intended change or an explicitly paired switch.
3. Use one matched pair for smoke. Three alternating headline pairs plus the
   4,096-token cross-disciplinary screen are internal qualification only.
   Promotion starts at the frozen 5/7/9 formal looks and adds the final matched
   context screen at 8,192 tokens or longer plus both DeepSeek regressions.
   Long-context runs use the ordinary no-DSpark control, numerical checks, and
   mandatory zero-fallback RDMA proof.
   Call `scripts/candidate-gate.py begin-pair` before either arm of every
   headline pair. Set `DS4_BENCH_CANDIDATE_ID` for both headline launches; the
   launcher starts the remote supervisor, records its generated run ID and
   declared order before coordinator inference, and refuses a duplicate arm,
   mislabeled order, or arm that contradicts the frozen AB/BA order. Afterward,
   `record-result` seals the CSV state, manifest, both logs, both statuses, and
   completion/cleanup attestations before another arm may start. The producer
   flushes its completion attestation before exposing the timing row. Before
   baseline or candidate launch, its executable must report the SHA-256 of the
   live `ds4_bench.c`; genesis and promotion independently resolve the
   committed bytes. A stale producer is ineligible and consumes no formal arm.
   Finish both arms before beginning the next pair. Invalidate only the latest
   pair when its first journaled arm failed before producing a complete result;
   once the second arm is journaled the pair cannot be replaced. Name the exact
   run ID and retain its manifest, both rank logs, coordinator and worker
   statuses, the adjacent result CSV's absence or incomplete bytes, and the
   producer/launcher completion attestations. Both statuses must be nonsignal
   and at least one must be nonzero. The sole signal exception is an attested
   launcher cleanup `TERM` on the worker behind a nonsignal nonzero coordinator
   status. A completed 300-token timing row or completion attestation cannot be
   invalidated. A valid
   slow result remains in the sequence. A pair index has at most two attempts and a
   candidate has at most two invalidations total; promotion discloses the
   count plus every earlier same-lane formal candidate, with exact-switch and
   available binary-match flags. Closing a candidate does not reset its formal sequence: a source
   commit can initialize only one formal candidate. A repaired or distinct
   hypothesis needs a new commit and candidate ID.
   A genuine infrastructure failure after the second arm is journaled closes
   that candidate; recovery starts at a new source commit and repeats the formal
   sequence. This cost is intentional because a selective second-arm rerun would
   condition the replacement on an already observed first-arm result.
4. Compare manifests with `scripts/compare-bench-manifests.py` and apply the
   candidate-gate, raw-log, CSV, and both-status validators. A manifest check
   does not prove build provenance, numerical correctness, or promotion.
5. Preserve failed runs beside successful runs and bind every reported result
   to its source and frozen artifact.

Repeated-Student promotion additionally requires the active baseline's
machine-recomputed bank of nine alternating control/control pairs under the
same source, model, binary, workload, effective environment, and RoCE v2 route.
This bank is reusable across candidates but cannot reuse candidate artifacts.
If its variance, drift, order/label bias, skew, or residual checks fail, use the
predeclared fixed-nine exact-sign path.

## Completion

Run targeted CPU/protocol checks and
`scripts/check-research-root-contract.sh` before committing. Keep candidate
kernels default-off until their applicable gates pass. Keep advisor reviews in
the promotion dossier; a repair or experiment commit is not a promotion.

## Active uncensored-model verifier track

`research/glm53-uncensored-six-kda-20260917` extends current main with
default-off exact BF16 reuse and accepted-prefix target verification for the
unchanged uncensored Q4 GGUF. Its dossier is
`$DS4_RESEARCH_ROOT/candidates/glm53-uncensored-20260915/`. Local echo-peer
checks do not establish real TP throughput or promotion. Session publication
requires coordinator agreement on both ranks' verification and commit outcomes.

The explicit native block45 draft step uses its own compact MLA state and a
bounded eight-expert window of original Q4_K bytes. Its workspace remains bound
to one state, immutable model and TP link. Native residual/norm semantics adapt
donors `fa09e19` and `519cf21` from the retained September 14 successor; they do
not merge that track. The adaptation, primary-source references and test scope
are recorded in the same dossier under `research/native-draft-step-plan-20260918.md`.
This step alone does not enable session drafting or establish acceptance/speed.

The default-off `DS4_GLM5_NATIVE_DRAFT=2/4/6/8` session experiment connects native
drafting, exact target verification and accepted-prefix refresh. Both ranks
negotiate the width and agree before history publication; failures invalidate
both states. Socket and controlled-arithmetic tests exercise orchestration,
including 8K frontiers, but do not establish actual RoCE, 8K model quality,
representative acceptance or whole-model speed. See the same dossier's
`research/native-session-plan-91508e1.md` and associated result report. Do not
enable this experiment for deployment before those checks and advisor review.

Width 6 means six target rows (root plus five proposals), with a distinct
canonical hello bit41 and exact M6 BF16/Q8 dispatch. It retains the existing
eight-expert native window and allocates only M2/M4/M6 verifier workspaces.
Short generation tails select 4/2/1 explicitly. Mixed width encodings and
workspace/configuration mismatch refuse before drafting; no rounding to M8.
See `research/native-width6-plan-3fae014.md` in the same dossier for validation.

`DS4_TP_BULK_RECV_READY=1` is a separate default-off transport experiment.
It negotiates hello bit39 and confirms both mlx5 receive queues before bulk
sends, retaining the registered slab and RDMA payload path. The small-payload
RoCE fixture is `tests/test_tp_bulk_small`; model evidence and protocol review
are still required. See `research/bulk-receive-ready-plan-3bfefc5.md` in the
same candidate dossier.

`DS4_GLM5_NATIVE_WINDOW_REUSE=1` separately retains matching original Q4_K
expert entries in the native workspace's existing eight-slot window. Its
54MiB capacity is unchanged; generic trunk window rebinding is unaffected.
This remains default-off, with native owner/model/rank binding and terminal
consumer synchronization required. The rationale and validation scope are in
`research/native-window-reuse-plan-48bb294.md` in the same dossier.

`DS4_ROCM_GLM5_VERIFY_SHARED_Q8=1` is a default-off exact shared-expert
batch experiment for resident KDA routed trunk layers in the target verifier.
It retains scalar mHC prefixes and original Q8_0 bytes, then shares weights
across M2/4/8 using existing activation scratch. Router agreement, routed Q4_K
evaluation and TP exchanges remain per token. Unsupported shared-pair modes
or nonresident layouts refuse. MLA and ordinary decode are unchanged by the
selector. See `research/shared-q8-integration-plan-d02a636.md` for the bounded
hypothesis and required real-weight, network and promotion checks.

`DS4_ROCM_GLM5_VERIFY_FFN_HANDOFF=1` separately batches route readbacks,
agreements and FFN payload exchanges across M2/4/8 in resident KDA routed
verification. It retains scalar router/expert arithmetic and existing scratch,
requires shared-Q8 batching and negotiated bit40, and stays default-off.
Layer agreements carry failure status with bounded I/O; negotiated bulk header
waits also have deadlines and terminate both channels on failure. The optional
`DS4_GLM5_NATIVE_PHASE_PROFILE=1` reports completed native-cycle boundaries
without extra GPU fences. See `research/handoff-plan-927ae0e.md` in the same
dossier. Echo-peer component fixtures and protocol tests do not establish
whole-model speed, real RoCE correctness or promotion.

`DS4_ROCM_GLM5_VERIFY_MLA_FFN_HANDOFF=1` extends the native shared-Q8/FFN
schedule to resident MLA trunk layers. It requires the existing handoff and
shared-Q8 settings, a valid native width, and negotiated bit42. It preserves
serial causal attention and saves each residual in existing batch scratch
before the shared projection and one FFN handoff. Prefill, native drafting,
model bytes and expert-window capacity are unchanged. Selector/configuration
mismatch refuses. This is default-off Lane A research; the plan and measured
results belong in `research/mla-ffn-handoff-plan-710f97e.md` in the dossier.

`DS4_ROCM_GLM5_VERIFY_FFN_QUEUE=1` queues the existing resident M1 expert
kernels and activation copies across verifier rows on stream0, with one
completion fence before phase1 status agreement. It requires native drafting,
FFN handoff, and negotiated bit43; malformed or mismatched settings refuse.
The default-off control completes each row separately. No prefill, model,
weight allocation or native draft-window change is involved. Fixture-only
HIP event wrappers measure packed expert device intervals separately from
complete loop wall time; they are not linked into model executables. The plan
is `research/ffn-queue-plan-d65372d.md` in the candidate dossier.
