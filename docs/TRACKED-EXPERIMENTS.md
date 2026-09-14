# Tracked DS4 experiments

This workflow covers ordinary inference, transport, kernel, and harness
research. DSpark-specific checks apply only when a DSpark candidate is in
scope. A local checkpoint records work; it is not permission to merge or push.

For this GLM5.3 Flash goal, the user explicitly requires Antirez's original
GGUF with unchanged weights and quantization (2026-09-13). Do not create or
substitute a compact/requantized model or requantize weights at runtime.
The compact-model tracks are paused and retained unmerged. Continue improving
the original-model kernels, scheduling and mandatory RoCE v2 transport.

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
3. Use the documented production workload for headline timing and retain the
   required repeated runs, the 4,096-token cross-disciplinary screen, and a
   final matched context screen at 8,192 tokens or longer. Both long-context
   runs use the ordinary no-DSpark control, numerical checks, and mandatory
   zero-fallback RDMA proof.
4. Compare manifests with `scripts/compare-bench-manifests.py` and apply the
   candidate-gate, raw-log, CSV, and worker-status validators. A manifest check
   does not prove build provenance, numerical correctness, or promotion.
5. Preserve failed runs beside successful runs and bind every reported result
   to its source and frozen artifact.

## Completion

Run targeted CPU/protocol checks and
`scripts/check-research-root-contract.sh` before committing. Keep candidate
kernels default-off until their applicable gates pass. Keep advisor reviews in
the promotion dossier; a repair or experiment commit is not a promotion.
