# regextree

Interpretable text categorization using LLM-generated regex patterns as weak learners in a gradient boosting (or decision tree) framework.

## Install

```bash
cd /Users/jsekar/repos/regextree
source .venv/bin/activate
pip install -e ".[dev]"
```

Requires `GEMINI_API_KEY` in environment (or pass `api_key=` directly). A `.env` file is included locally.

## Quick Start

```python
import regextree

model = regextree.RegexBoostClassifier(
    pool_size=20,
    step_count=5,
    eta=0.2,
    llm="gemini",
)
model.fit(x_train, y_train, eval_set={"val": (x_val, y_val)})

y_pred = model.predict(x_test)

# Interpretability
model.patterns()       # {rule_id: r"...", ...}
model.description()    # {rule_id: "human description", ...}
model.rules()          # List[Rule] with .regex, .target_class, .weight
match_df = model.matches(x_test)  # DataFrame of boolean pattern matches
```

## How It Works

Each boosting round:
1. Computes per-sample gradients (softmax cross-entropy)
2. Identifies the hardest class (highest mean negative gradient)
3. Samples hard positives/negatives weighted by gradient magnitude
4. Prompts Gemini for `pool_size` candidate regex patterns
5. Evaluates candidates by gradient-weighted information gain
6. Selects the best pattern and adds it as a weighted rule

Final prediction: sum rule weights per class for matching patterns, argmax.

## Architecture

| Module | Role |
|--------|------|
| `boost.py` | `RegexBoostClassifier` — gradient-boosted ensemble of regex rules |
| `tree.py` | `RegexTreeClassifier` — decision tree with regex splits (Gini impurity) |
| `rule.py` | `Rule` dataclass + matching engine + logit accumulation |
| `scoring.py` | Softmax, cross-entropy, gradients, candidate scoring |
| `prompt.py` | Prompt templates, LLM response parsing, description generation |
| `display.py` | `.patterns()`, `.description()`, `.matches()`, `.rules()` mixin |
| `llm/` | Abstract `LLMClient` protocol + Gemini 2.5 Flash implementation |

## Eval Results (step_cluster.feather, one-vs-rest)

42,586 math steps across 94 clusters. 5 boosting steps per cluster, Gemini 2.5 Flash (no thinking), ~7s per cluster.

| Cluster | n_pos | Accuracy | F1 | Precision | Recall |
|---------|-------|----------|------|-----------|--------|
| 4 | 1556 | 0.900 | 0.341 | 0.225 | 0.707 |
| 1 | 1548 | 0.881 | 0.336 | 0.211 | 0.829 |
| 64 | 328 | 0.875 | 0.098 | 0.052 | 0.879 |
| 83 | 228 | 0.938 | 0.145 | 0.079 | 0.978 |
| 46 | 438 | 0.247 | 0.026 | 0.013 | 0.989 |

**Mean F1: 0.19** — high recall, low precision. Regexes are too broad without domain separation (structurally similar clusters like 64/83 share patterns).

## Known Issues / Next Steps

- **Domain split needed**: splitting data by math domain before fitting should dramatically improve precision by removing confusers from the negative pool
- **Refinement prompts**: show the LLM its false positives and ask it to tighten the regex
- **Precision thresholds**: reject candidates that fire on >X% of negatives
- **More boosting steps**: 5 steps is conservative; 10-20 may help with fine-grained clusters
