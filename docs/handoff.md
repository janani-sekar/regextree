# Handoff: regextree / regextree-agent against the discovery pipeline

Status of the two libraries as of this commit, read against the misconception discovery
README (the seven-phase pipeline). "Adapting discovery to $Q$, $R$" is the design proposal for
the discovery redesign — nothing in that section is built yet. Two decisions are needed before
Phase 7 output can be trusted; they are at the bottom.

## What the libraries do today

`regextree` (library):

- `ManipulationRule` with candidate / realized / false-positive semantics matching `s32`,
  including the canonical sympy expand-and-compare used there.
- `evaluate_rule_on_transitions` — one rule against a binary label column.
- `load_rule_from_module` — load an agent's `exports/results.py`, either shape.
- `ManipulationBank` — persist discovered manipulations with provenance and stats. **New.**
- `label_transitions` / `summarize_labels` — full-dataset (transition × manipulation)
  labeling. **New.**
- Regex boosting for step groups, with LASSO weight refit.

`regextree-agent` (service):

- Minimal Docker image, headless Claude, `run_claude_docker` + `discover_manipulations_claude`.
- Prompt `discover_v1`, scoring harness bind-mounted read-only into the sandbox.
- `01_score_baseline.py` (hand specs vs clusters), `02_discover_cluster.py` (one cluster).
- One live run: `linear_equations` cluster 6, val precision ≈ 0.26, F1 ≈ 0.07.

## Answers to the questions in the pipeline README

**Is the existing manipulation bank required as input?** No. Only `01_score_baseline.py`
reads `misconceptions.py`, and that is a standalone baseline CLI, not a dependency of
discovery. The pipeline can run with no pre-existing bank, as preferred.

**What inputs do the libraries need?** For discovery: a dataframe with `domain`,
`step_prev_expression`, `step_expression`, `transition_correct`, and a label column. For
Phase 7 labeling: `transition_id`, parsed `prev_expr` / `curr_expr`, `step_correct`.
Expression strings become sympy via `efm.Parser`, so the efm source tree must be on the
machine (see below).

**Where the libraries live:** `~/repos/regextree` and `~/repos/regextree-agent`, not
`~/repos/efm-discovery`. Worth fixing in the pipeline scripts' paths.

## Fixed in this pass

**Hardcoded efm path.** `regextree-agent/parse.py` pointed at
`/Users/jsekar/repos/efm/src` and loaded the parser at module import, so `import discover`
raised on any other machine. It now resolves `$EFM_SRC`, falls back to `~/repos/efm/src`,
loads lazily, and raises a message naming the variable to set. `efm_available()` is
exported so eval scripts and tests can degrade cleanly.

**Manipulation bank (Phase 6 output).** Phase 6 says discovery "saves the final outputs to
a bank of manipulations, along with basic performance statistics" — there was no such
thing. `ManipulationBank` is a directory plus a JSONL manifest. Because a rule carries a
live `apply_fn` and cannot be pickled meaningfully, the bank stores the exported module
source verbatim and reloads through `load_rule_from_module`; every banked rule stays
readable and re-runnable. Ids are `<name>_<8 char source hash>`, so identical re-adds
collapse instead of duplicating. Provenance is free-form, which is where the winning
$(Q, R)$ pair and hyperparameters belong.

**Phase 7 labeling.** `label_transitions(df, bank.rules())` emits the long table, one row
per (transition, manipulation). Rule outputs are cached per distinct previous expression,
so cost tracks unique expressions rather than rows — 6k transitions × 4 manipulations runs
in about 0.3s, so the full observed table against a real bank is not a concern.

The table carries `manipulation_candidate` and `manipulation_false_positive` in addition to
`manipulation_realized` (see decision 2).

## Still open

**1. Discovery is 1-vs-all, but the pipeline now wants a family.** The additional notes say
the two-stage decomposition leans away from 1-vs-all discovery and toward finding a $K_2$
manipulation family simultaneously. Today `discover_manipulations_claude` takes a single
`cluster_id`, the agent writes exactly one `exports/results.py`, and
`DiscoveredManipulation` holds one rule. This needs a multi-rule export contract and
family-level scoring — coverage and assignment across $K_2$ rules rather than one rule's
F1. Not started.

**2. The agent still sees binary labels, not the three-way split.** The notes ask for
irrelevant transitions, transitions in step group $Q$ but not response group $R$, and
transitions in $R$, so the agent can hypothesize the candidate rule and the realized rule
at once. `_prepare_labeled_frame` collapses everything to one 0/1 `label`, and
`prompts/discover_v1.txt` tells Claude that `label=1` means cluster membership. Scoring
needs to follow: candidate rule against $Q$ membership, realized rule against $R$
membership, reported separately. There is currently no candidate-vs-$Q$ scoring at all —
`evaluate_rule_on_transitions` scores `candidate ∧ realized` against a single column and
treats candidate coverage as descriptive only. Not started.

**3. `regextree-agent/WRITEUP.md` is stale.** Its "what we still need from you" asks for
counterfactual id and student cluster id so the library can recompute `mc_r2_partial`. The
new pipeline puts that scoring in R (phases 2, 4, and final eval), so the library needs
`transition_observed_df` with $Q$ / $R$ membership and `realized` attached, and nothing
about student types.

## Adapting discovery to $Q$, $R$

Items 1 and 2 above are one redesign, and the $Q$, $R$ structure maps onto the library's
existing semantics closely enough to drive it. The pipeline README already says as much: a
step group is "an analogue for a descriptive candidate rule" and its response sub-groups are
"analogues for realized rules." A single `apply_fn` induces exactly that pair:

| Pipeline object | Library object | Rule half |
|---|---|---|
| $q \in Q$ (question-subset) | `prev_expr` | candidate: `apply_fn(prev) != []` |
| $(q, r) \in R_{k_2}(Q)$ | `(prev_expr, curr_expr)` | realized: `canonical(curr) in apply_fn(prev)` |

So one manipulation should be scored against two targets rather than one collapsed label, and
the redesign is mostly about making that explicit end to end: in the data handed to the agent,
in the metrics, in the selection rule, and in the export contract.

### What the agent sees

**Indicator columns, not an ordinal.** A single three-level label (0 irrelevant, 1 in $Q$ not
$R$, 2 in $R$) is enough for one rule but breaks for a family: $\tau_2$ thresholding lets a
transition load onto several response groups at once, so the target is multi-label, not
multi-class. The workspace frame should carry `in_q` plus one boolean per response group,
`in_r_1 … in_r_K2`, with the three buckets derived from those. The single-rule case is then
just $K_2 = 1$.

**The three buckets and what each one teaches.** Irrelevant rows ($q \notin Q$) are the only
evidence about applicability. Rows with $q \in Q$ but $(q, r) \notin R_k$ are the transitions
where the student could have made the error and did not — they teach the realized rule, and
they are also the comparison group the psychometric metric regresses over. Rows in $R_k$ are
the positives.

**Phase 6 as written cannot identify the candidate rule.** The phase body says to "filter to
all included transitions from $Q$," which makes every row in the agent's sample a candidate by
construction: no applicability signal at all, only realization. The additional note asking for
irrelevant transitions alongside the other two buckets is therefore load-bearing rather than a
refinement, and the two passages should be reconciled in the pipeline README.

**Sampling.** The current downsampler keeps all positives and randomly samples negatives,
which would gut the $Q$-not-$R$ bucket. Use stratified quotas across all $K_2 + 2$ buckets
instead. For the irrelevant bucket, prefer structurally *near* misses over uniform draws —
uniformly sampled negatives make applicability look trivially separable and teach a lazy
candidate rule. The boost track already has this idea as `subgroup_labels`, which prefers
structurally similar confusers in negative sampling; the same principle applies here.

**Split by question, not by row.** `prepare_discovery_workspace` currently does a row-wise
stratified `train_test_split`. Many observed transitions share a $q$, so a rule that memorizes
particular expressions sees them again in validation. Since $Q$ is a question-subset and the
question is whether applicability generalizes to *unseen* questions in the group, the split
should be grouped on $q$.

### Metrics

Two independent pairs, reported separately, never averaged into one number:

- **Candidate agreement** — `is_candidate(q)` against $q \in Q$, computed over **distinct
  questions**.
- **Conditional realized agreement** — `is_realized(q, r)` against $(q, r) \in R_k$, computed
  over **distinct $(q, r)$ pairs restricted to $q \in Q$**.
- **Leakage** — realized matches where $q \notin Q$. Should be near zero if applicability is
  tight; a nonzero value means the rule is firing outside its own step group.

The reason to split them: today a rule with the right applicability and wrong outputs scores
the same low F1 as a rule with wrong applicability, so the agent gets no signal about which
half to fix.

**Mind the units.** $Q$ and $R$ are sets of distinct questions and distinct pairs, while the
Phase 4 size guards are in observed counts (100 observed steps and 20 observed transitions;
ideally 500 and 100). So structural agreement should be unweighted over unique $q$ and unique
$(q, r)$ — otherwise frequently-attempted items dominate the candidate metric — while size
gates and anything meant to approximate the psychometric metric use observed rows.

### Selection, and the real acceptance test

**Selection rule.** Maximize conditional realized F1 subject to a floor on candidate recall,
with the Phase 4 size thresholds as hard gates. The asymmetry is deliberate: the candidate rule
defines the subpopulation the final regression runs on, so a too-narrow candidate rule starves
the sample, while a too-broad one only dilutes the realized rate.

**But agreement F1 is a proxy.** What we actually want is that the manipulation preserves the
type-dependence the black-box $(Q, R)$ pair was selected for, and that is directly checkable.
Recompute the Phase 4 primary score with $\hat{R}$, the set of $(q, r)$ pairs the manipulation
marks realized, substituted for $R$:

$$ y_s = \frac{\sum_{(q, r) \in \hat{R}} P(r \mid q, s)}{\sum_{(q, r) \in Q} P(r \mid q, s)}, $$

then run the same baseline and type-factor regressions and compare `partial_r2 *
resid_var_baseline` against the black-box original. If the score survives substitution, the rule
captured the misconception. If it collapses, the rule matched rows and lost the signal, which F1
alone will not reveal. This is the natural bridge from Phase 6 back to Phase 4, and the right
metric to gate bank admission on.

It needs the counterfactual $P(r \mid q, s)$ matrix, which the libraries do not currently
receive. It is also where manipulations beat the black-box label rather than merely imitating
it: an `apply_fn` can label enumerated counterfactual responses that never appeared in the
observed data, so $\hat{R}$ can cover response candidates $R$ could not see. That argument is
worth making in the proposal even if the check is not implemented.

### Degenerate solutions to guard against

**Inflate the output set.** The cheapest way to maximize realized recall is an `apply_fn` that
returns every algebraically reachable expression, making realization near-certain inside $Q$.
Conditional realized *precision* catches this, but only if it is reported alongside recall.
Logging output cardinality per candidate makes the failure visible rather than inferred.

**Memorize expressions.** Handled by grouping the split on $q$; without it, this is the path of
least resistance for an agent that can read `train.csv`.

**Match $Q$ by surface features.** A rule can hit $Q$ using incidental text properties rather
than a real structural precondition. Generalization to unseen $q$ in the same group is the check,
which is another reason the grouped split matters more here than under 1-vs-all.

**An unattainable target.** Phase 3 says the response decomposition does *not* restrict to
incorrect transitions, so $R$ can contain correct transitions. If `is_realized` keeps gating on
`step_correct`, the agent is being scored against a target it structurally cannot hit on those
rows. Decision 1 below is therefore not only about the Phase 7 export — it decides whether the
discovery objective is well-posed at all.

### Families of $K_2$ rules

**No new rule type is needed.** All $K_2$ response groups are decomposed within the same $Q$, so
they share a step group and differ only in how the student went wrong: one applicability region,
several wrong-output generators. Since the pipeline README defines a misconception as a single
application function, a family is just a set of ordinary `ManipulationRule`s sharing a provenance
tag, which `ManipulationBank` already supports. Encourage the shared candidate structure in the
prompt and *measure* whether it held rather than enforcing it in the type.

**Family metrics.** Per-rule conditional realized F1 against its own $R_{k_2}$; family coverage,
meaning the share of $\bigcup_k R_k$ realized by at least one rule; per-rule purity; and a
cross-talk matrix of rule $i$ realized on $R_j$. Cross-talk is the sharpest diagnostic, because
Phase 5 has already greedily dropped $(Q_0, R)$ pairs whose $R$ overlaps a chosen $R_0$ — the
family handed to the agent is deduplicated by construction, so high off-diagonal realization
means the manipulations collapsed distinctions the black-box decomposition had kept.

**This can be built on `label_transitions` rather than new matching code.** That function already
emits per (transition, rule) candidate and realized flags for an arbitrary set of rules; joining
its output to the `in_r_*` indicator columns gives coverage, purity, and the full cross-talk
matrix directly. The Phase 7 labeler and the family scorer are the same computation at different
scopes.

### Export contract and score parity

**One module per response group.** Keep `exports/results_<group>.py`, each with the existing
`apply_fn` plus `NAME` / `DESCRIPTION` shape, and add an `exports/family.json` mapping response
groups to modules. That leaves `load_rule_from_module` and the bank's one-module-per-manipulation
invariant untouched; a single module holding $K_2$ rules would duplicate source across bank
entries. Bank provenance should then carry `step_group`, `response_group`, and a shared
`family_id` so a family can be reconstructed from the manifest.

**The container scorer must match the host scorer.** `container_assets/scoring/` is a hand-copied
subset of the library's matching logic, and the agent optimizes whatever *that* reports. Under a
two-part objective the drift risk stops being cosmetic: if the container still prints a single F1
while the host selects on candidate recall plus conditional realized F1, the agent is tuning the
wrong thing. Rather than maintaining a parallel copy, generate the container scoring directory
from the installed package at workspace-prep time.

### Where the regex track fits

Since $Q$ is a set of questions and questions are text, `RegexBoostClassifier` can characterize
$Q$ directly as a handful of interpretable patterns over previous expressions — which is what
"descriptive candidate rule" is asking for. Use it either to seed the agent prompt with an
applicability hint, or as a fallback candidate rule where the sympy precondition resists clean
expression. This is the first job the $Q$, $R$ structure creates for the regex half of the
library, which today does not interact with the manipulation track at all.

### Prompt and iteration order

Tell the agent to fix applicability first and outputs second. The two halves are coupled through
one function — broadening applicability adds outputs, which adds spurious realized matches — so
iterating on outputs before the candidate region is settled is wasted effort. It also needs to be
told what the middle bucket *means* (could have made this error, did not) and that the response
groups are siblings carving up one step group, since that is the whole reason to discover them
together.

## Decisions needed

**1. Can a correct step realize a manipulation?** `is_realized` returns `False` whenever
`step_correct` is True, which is the `s32` convention and what the library has always
advertised. But Phase 3 says the response decomposition does *not* restrict to incorrect
transitions, and the misconception definition in the pipeline README defines "realized"
purely structurally: did the manipulation appear in the current expression. So an $R$ that
contains correct transitions will silently never be marked realized.

`label_transitions` exposes `realized_requires_incorrect`, defaulting to `True` to preserve
current behavior. If the structural reading is what Phase 7 should export, flip it to
`False` at the call site — no library change needed. Flagging rather than choosing, since it
changes what the Phase 7 table means.

**2. Should the Phase 7 export carry the candidate flag?** The stated schema is
`transition_id, manipulation_id, manipulation_realized`, but the evaluation section
regresses over "all transitions where the previous expression is a candidate." With
`realized` alone the downstream R code cannot construct that subpopulation. The label table
therefore includes `manipulation_candidate` and `manipulation_false_positive`; drop the
extra columns on export if the narrower schema is wanted.
