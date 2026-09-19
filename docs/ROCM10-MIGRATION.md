# ROCm 10 migration acceptance

On 2026-09-19 the user approved a one-time exception for the failed DeepSeek
Q2/Q4 and GLM Q2 quality checks and authorized merging
`promotion/glm53-mtp-20260919`. The approved implementation is
`a0316862571c0a374cfd39abca0c66cf97fa3561`; subsequent acceptance documentation
does not change inference, settings, binaries, models, or measured throughput.

Quality-gate status for those three comparisons is **overridden by user
exception, not passed**. The failed results and immutable pre-migration quality
anchor are retained. Standing thresholds and future promotion requirements
are unchanged. Repeated timing was separately waived; the performance tables
retain individual observations and their original source/SDK labels.

| Fixed-reference quality comparison | Cases / target tokens | Perplexity change | Approximate 95% interval |
|---|---:|---:|---:|
| Huihui DeepSeek Q2 | 100 / 2,289 | +0.4477% | -0.40% to +1.34% |
| Antirez DeepSeek Q4 | 100 / 2,289 | +0.5305% | -0.14% to +1.22% |
| Antirez GLM Q2 | 100 / 11,559 | +0.1069% | -0.20% to +0.39% |

Lower perplexity is better. The intervals resample whole cases and permit
worsening; they do not establish positive quality drift or non-inferiority.
The existing score screens also fail individual-case limits. Some token
changes occur at large prediction margins, and GLM Q2 has no usable API top-one
labels. Short fixed-reference tests do not establish long-prefill or compact
top-two equivalence. Matched SDK comparisons do not independently validate
every old-to-new deployed setting change.

The release pins ROCm Core SDK 10.0.0 and preserves ordinary decode as the
default. Native MTP and DSpark remain opt-in; this migration does not promote
their performance or enable them in deployment. Original weights and
quantization are unchanged. RoCE v2 runs retain zero payload fallback and no
expanded-weight cache. DeepSeek Q2's ordered-prefill table recipe remains an
explicit opt-in, not a new launcher default.

Completed validation includes the CPU/protocol and quality-harness suites,
matching executable/SDK identities on both nodes, GLM Q4 4K/8K screens, and an
isolated HTTP smoke with clean rank exits. A quality-harness test pass does not
change the failed model-quality verdicts. The prior main revision `474a1a4`
and frozen ROCm 7.14 artifacts remain available for rollback.

Full failed results, advisor reviews, exception authorization, build identities,
and deployment evidence are retained under
`$DS4_RESEARCH_ROOT/candidates/glm53-mtp-20260919/`, including
`rocm10-quality-exception-20260919.md` and
`rocm10-tracked-quality-caps-results.md`. The prepared deployment preserves the
existing endpoint, port, and key; a source merge does not start that service.
