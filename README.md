# regextree

Interpretable text categorization for math education research. The library provides two complementary approaches:

1. **Regex boosting** — LLM-generated regex patterns as weak learners in a gradient boosting framework (step-level clustering).
2. **Manipulation rules** — sympy-based candidate/realized matching for transition-level misconception discovery.

For Claude-agent-based discovery (Docker orchestration, prompts, eval loops), see the companion project **[regextree-agent](../regextree-agent/)** (`01_score_baseline.py` → `02_discover_cluster.py`; see that repo’s `WRITEUP.md` for James).

## Install

```bash
cd regextree
source .venv/bin/activate
pip install -e ".[dev]"
```

Regex boosting requires `GEMINI_API_KEY` in environment (or pass `api_key=` directly).

## Manipulation rules

A manipulation rule predicts student errors on expression transitions:

- **Candidate** — `apply_fn(prev_expr)` returns a non-empty list of plausible wrong next expressions
- **Realized** — the observed next expression matches one of those outputs (incorrect steps only)
- **Predict** — candidate and realized (used for 1-vs-all cluster label matching)

This mirrors the logic in `s32_evaluate_manipulations.py`.

```python
import sympy
from regextree import ManipulationRule, evaluate_rule_on_transitions

def apply_sign(prev_expr):
    if not isinstance(prev_expr, sympy.Equality):
        return []
    lhs, rhs = prev_expr.args
    if not lhs.is_Add:
        return []
    outputs = []
    for arg in lhs.args:
        if arg.is_Symbol:
            continue
        remaining = [a for a in lhs.args if a != arg]
        new_lhs = sympy.Add(*remaining, evaluate=False) if len(remaining) > 1 else remaining[0]
        new_rhs = sympy.Add(arg, rhs, evaluate=False)
        outputs.append(sympy.Eq(new_lhs, new_rhs, evaluate=False))
    return outputs

rule = ManipulationRule(
    name="sign_error",
    description="Term moved across '=' without sign change.",
    apply_fn=apply_sign,
)

prev = sympy.Eq(sympy.Symbol("x") + 3, 12, evaluate=False)
curr = sympy.Eq(sympy.Symbol("x"), 15, evaluate=False)

rule.candidate(prev)   # True
rule.realized(prev, curr, step_correct=False)  # True
rule.predict(prev, curr, step_correct=False)   # True
```

### Scoring against cluster labels

The package scores how well a rule's predictions match binary cluster labels. Out-of-sample ANOVA / incremental variance evaluation (student type beyond ability + difficulty) is the user's responsibility.

```python
metrics = evaluate_rule_on_transitions(
    df,
    rule,
    prev_expr_col="prev_expr",
    curr_expr_col="curr_expr",
    label_col="label",              # 1 = in target transition cluster
    step_correct_col="step_correct",  # optional, for false-positive rate
)
# metrics: precision, recall, f1, n_candidate, false_positive_rate, ...
```

### Loading discovered rules

External agents write a Python module with a `rule` object. The library loads it deterministically:

```python
from regextree import load_rule_from_module

rule = load_rule_from_module("exports/results.py")
```

Expected export format:

```python
# exports/results.py
from regextree import ManipulationRule

def apply_my_manipulation(prev_expr):
    ...

rule = ManipulationRule(name="...", description="...", apply_fn=apply_my_manipulation)
```

## Regex boosting

```python
import regextree

model = regextree.RegexBoostClassifier(
    pool_size=20,
    step_count=5,
    eta=0.2,
    llm="gemini",
    objective="precision",              # or "f1", "recall", "cross_entropy"
    max_negative_match_rate=0.05,
    weight_method="lasso",              # or "gradient"
)
model.fit(x_train, y_train, eval_set={"val": (x_val, y_val)})

y_pred = model.predict(x_test)
model.patterns()
model.rules()
```

The `.fit()` loop is a convenience wrapper over three explicit phases:

```python
round_data = model.generate_round(x, labels, logits, step=0)  # LLM call
selected = model.select_candidate(round_data, x, labels, logits)
rule = model.add_rule(selected, step=0)
model.refit_weights(x, labels)  # when weight_method="lasso"
```

Pass `subgroup_labels=` (e.g. cluster IDs) to prefer structurally similar confusers in negative sampling.

## Architecture

| Module | Role |
|--------|------|
| `manipulation/` | `ManipulationRule`, candidate/realized matching, cluster-label scoring, rule loading |
| `boost.py` | `RegexBoostClassifier` — gradient-boosted ensemble of regex rules |
| `tree.py` | `RegexTreeClassifier` — decision tree with regex splits |
| `rule.py` | Regex `Rule` dataclass + logit accumulation |
| `scoring.py` | Softmax, cross-entropy, gradients, objective functions |
| `lasso.py` | L1 weight refit on regex match matrix |
| `prompt.py` | Prompt templates for regex generation |
| `display.py` | `.patterns()`, `.description()`, `.matches()`, `.rules()` |
| `llm/` | `LLMClient` protocol + Gemini implementation |

## Package boundary

| In regextree (library) | In regextree-agent (service) |
|------------------------|------------------------------|
| Manipulation matching + scoring | Docker orchestration |
| Regex boosting | Claude agent prompts |
| Rule loading from Python modules | `discover_manipulations_claude()` |
| Unit tests | Eval scripts, domain × cluster loops |

## Tests

```bash
pytest tests/ -v
```
