"""One-vs-rest evaluation for regextree on step_cluster data."""

import os
import sys
import time

os.environ["GEMINI_API_KEY"] = "AIzaSyB_fiIiYzDLsfjCCR0hUj97GRdux_G2Q4o"

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, "/Users/jsekar/repos/regextree/src")
import regextree


CONFIGS = {
    "baseline": dict(
        objective="cross_entropy",
        max_negative_match_rate=None,
        weight_method="gradient",
    ),
    "precision": dict(
        objective="precision",
        max_negative_match_rate=0.05,
        weight_method="lasso",
        lasso_C=0.5,
        confuser_fraction=0.7,
    ),
}


def evaluate_cluster(
    df,
    cluster_id,
    config_name: str,
    config_kwargs: dict,
    pool_size: int = 10,
    step_count: int = 5,
    eta: float = 0.3,
    verbose: bool = False,
):
    y_binary = (df["cluster_idx"] == cluster_id).astype(int).values
    x_all = df["step_prev_text"].tolist()
    subgroups = df["cluster_idx"].values

    x_train, x_val, y_train, y_val, sg_train, _ = train_test_split(
        x_all,
        y_binary,
        subgroups,
        test_size=0.2,
        random_state=42,
        stratify=y_binary,
    )

    use_subgroups = config_name == "precision"
    fit_kwargs = {"subgroup_labels": sg_train} if use_subgroups else {}

    model = regextree.RegexBoostClassifier(
        pool_size=pool_size,
        step_count=step_count,
        eta=eta,
        llm="gemini",
        sample_size=20,
        verbose=verbose,
        **config_kwargs,
    )

    t0 = time.time()
    model.fit(x_train, y_train.tolist(), **fit_kwargs)
    elapsed = time.time() - t0

    y_pred = model.predict(x_val)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, zero_division=0)
    prec = precision_score(y_val, y_pred, zero_division=0)
    rec = recall_score(y_val, y_pred, zero_division=0)

    if verbose:
        print(
            f"  [{config_name}] acc={acc:.4f} f1={f1:.4f} prec={prec:.4f} "
            f"rec={rec:.4f} rules={len(model.rules())} ({elapsed:.1f}s)",
            flush=True,
        )
        for rule in model.rules():
            print(
                f"    {rule.id}: r'{rule.regex}' -> class {rule.target_class} "
                f"(w={rule.weight:.4f})",
                flush=True,
            )

    return {
        "config": config_name,
        "cluster_idx": cluster_id,
        "n_positive": int(y_binary.sum()),
        "accuracy": acc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
        "n_rules": len(model.rules()),
        "elapsed_s": elapsed,
    }


if __name__ == "__main__":
    df = pd.read_feather("/Users/jsekar/Downloads/step_cluster.feather")
    print(f"Data: {df.shape[0]} rows, {df['cluster_idx'].nunique()} clusters", flush=True)

    test_clusters = [4, 1, 64, 83, 46]
    results = []

    for cid in test_clusters:
        n_pos = (df["cluster_idx"] == cid).sum()
        print(f"\nCluster {cid} ({n_pos} positives total)", flush=True)
        for config_name, config_kwargs in CONFIGS.items():
            print(f"  running {config_name}...", flush=True)
            results.append(
                evaluate_cluster(
                    df,
                    cid,
                    config_name,
                    config_kwargs,
                    verbose=True,
                )
            )

    results_df = pd.DataFrame(results)

    print("\n\nPer-cluster comparison:", flush=True)
    for cid in test_clusters:
        subset = results_df[results_df["cluster_idx"] == cid]
        print(f"\nCluster {cid}:", flush=True)
        print(subset.to_string(index=False), flush=True)

    print("\n\nMean metrics by config:", flush=True)
    summary = results_df.groupby("config")[["accuracy", "f1", "precision", "recall", "n_rules"]].mean()
    print(summary.to_string(), flush=True)
