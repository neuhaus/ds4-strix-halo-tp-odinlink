# DS4 repository contract

## Research evidence

The repository never owns a root-level `research-results` path. All raw and
curated research evidence is written to the branch-independent directory named
by `DS4_RESEARCH_ROOT`, whose default is the sibling directory
`../research-results`.

- Do not create a worktree-local directory, symlink, gitlink, or tracked path
  named `research-results`.
- Do not force-add ignored research artifacts.
- Benchmark and deployment tools must resolve the canonical root through
  `scripts/ds4-research-root.sh`.
- The two TP nodes have independent filesystems. Use
  `DS4_PEER_RESEARCH_ROOT` when the peer canonical path differs.
- Compact research reports and archive manifests live only on the orphan Git
  branch `research/results-archive`. User-facing maintained material belongs in
  `README.md` or `docs/`.
- Never overwrite divergent evidence or delete a source before checksum
  verification.

Run `scripts/check-research-root-contract.sh` before committing.

Classify performance candidates and validate their immutable evidence dossier
with `scripts/candidate-gate.py`. The authoritative lane A/B/C policy is
`$DS4_RESEARCH_ROOT/policies/GATE-PROMOTION.md`.

## Performance candidates

Performance work must remain on a named research branch until its applicable
correctness, mandatory-RDMA, long-context, ordinary-regression, and repeated
timing gates pass. Raw evidence goes to `DS4_RESEARCH_ROOT`; only source and
maintained documentation are merged to `main`.

Research smoke runs guide experiments but cannot establish a promoted or
published statistic. Three alternating matched pairs provide only an internal
`consistent-direction` qualification; they are not a 95% result and cannot
merge. Formal promotion starts at five pairs and may stop at the predeclared
5/7/9 repeated-confidence looks. A miss at five or seven adds two pairs; a miss
at nine is `not-demonstrated`, not proof that the retained research lead has no
value. A README/table claim requires a passing formal look.

Qualification uses a matched 4,096+300 run over the frozen cross-disciplinary
prompt. Before promotion, every ordinary candidate must pass a final matched
8,192-token or longer context run with 300 generated tokens and the
same model, toolchain, TP layout, transport, and trajectory checks. These are
broad regression screens rather than extra headline estimates. A candidate
that has only the 4K screen is incomplete and cannot merge to `main`.

Performance margins and the formal method come from the active baseline;
targets, switches, pair order, and invalidation policy are journal-bound at
candidate initialization. Performance-only policy amendments are prospective,
payload-reviewed, and separate from numerical/quality calibration. They cannot
be adopted while an open candidate in the same scope exists.

The repeated-Student method requires one reusable, candidate-independent bank
of exactly nine alternating control/control pairs from the same clean source,
binary, model, RoCE v2 route, workload, and effective environment. The gate
reopens the raw CSVs, manifests, rank logs, and both rank statuses and rejects
order bias, label bias, excessive skew or residuals, drift, nonzero fallback,
or an expanded-weight cache. Otherwise the scope uses the fixed-nine exact-sign
method.

Before each headline pair, append its randomized identity and initialized order
with `candidate-gate.py begin-pair`. After the remote worker supervisor starts
successfully but before coordinator inference begins, the launcher appends the
exact run ID, order, and arm with `record-run`. After the run it appends
`record-result`, content-addressing the CSV state, manifest, both logs, both
statuses, and producer/launcher attestations. The producer attestation is
flushed before the timing row becomes observable. The next arm cannot start until
the preceding result is sealed. Only a first-arm failure before a complete
timing result may replace the latest pair. Both statuses must be nonsignal and
at least one must be nonzero, except that an attested launcher cleanup `TERM`
on the worker is allowed only behind a nonsignal nonzero coordinator status.
Once the second arm is journaled, the pair is immutable; a valid slow result cannot be
discarded. Each pair index permits at most two attempts, each candidate permits
at most two invalidations total, and the published promotion record discloses
the count. A source commit can
initialize only one formal candidate, including after a close or failed look;
a new hypothesis or repaired attempt requires a new commit. Baseline genesis
and amendment require a clean checkout and reviewer-visible hashes for every
verdict-contributing verifier, including the gate orchestrator, controls,
benchmark producer, benchmark and quality launchers, worker supervisor,
research-root resolver, and GGUF type inspector. GLM-5.3 Flash promotion is
fixed to RoCE v2 with zero payload fallback.

Before baseline or candidate inference, remote worker launch, or `record-run`,
the launcher must verify that the
benchmark executable's embedded producer-source SHA-256 matches the live
`ds4_bench.c`. Promotion proof resolves the same identity from each run's
committed source. A stale or unbound benchmark executable is ineligible.
Baseline genesis independently resolves the committed producer bytes and
rejects missing, stale, or mixed benchmark executable identities.

## Research-track lifecycle

Named research branches are durable tracks rather than disposable release
candidates. A track may be marked `paused` or `given-up` when its hypothesis is
not worth pursuing; that status does not authorize merging it to `main`,
deleting its source, or treating its timing as a baseline. Preserve the branch,
its scoped commits, and its evidence so a later review can inspect it.

When a later direction finds a transferable idea, start a new named branch from
current `main`. Record the donor branch and commit, the exact adaptation (or
rejection), and the resulting patch hash in an adaptation ledger. Re-run the
current correctness, ordinary-regression, long-context, and mandatory-RDMA
gates for the new branch. Only that independently validated successor may be
merged; a direct merge or cherry-pick from a paused/given-up track is not a
promotion path.
