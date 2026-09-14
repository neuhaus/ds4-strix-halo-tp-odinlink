# Tracked DS4 experiments

Research work uses a named branch and stores all evidence beneath the canonical
`DS4_RESEARCH_ROOT`. Preserve unrelated dirty state and bind every benchmark to
its exact source, build, model, workload, transport, and effective rank
settings.

Before changing or timing a candidate, record its parent commit, hypothesis,
baseline, intended variable, correctness lane, and smallest falsifiable test.
Commit scoped source or harness work before publishable timing. Keep
experimental commits distinct from promotion to `main`.

Performance candidates remain on research branches until their correctness,
mandatory-RDMA, long-context, ordinary-regression, and repeated-timing gates
pass. Paused or given-up branches remain retained and unmerged; transfer a
useful idea through a fresh branch from current `main` and re-run the active
gates.

Run targeted tests and `scripts/check-research-root-contract.sh` before each
checkpoint. A diagnostic result or local commit is not permission to merge or
publish.
