"""One-vs-rest evaluation for regextree on step_cluster data."""

import os
import sys
import time

os.environ["GEMINI_API_KEY"] = "AIzaSyB_fiIiYzDLsfjCCR0hUj97GRdux_G2Q4o"

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, "/Users/jsekar/repos/regextree/src")
import regextree


def evaluate_cluster(df, cluster_id, pool_size=10, step_count=5, eta=0.3):
    y_binary = (df["cluster_idx"] == cluster_id).astype(int).values
    x_all = df["step_prev_text"].tolist()

    x_train, x_val, y_train, y_val = train_test_split(
        x_all, y_binary, test_size=0.2, random_state=42, stratify=y_binary
    )

    n_pos_train = sum(y_train)
    n_pos_val = sum(y_val)
    print(f"\nCluster {cluster_id}: {n_pos_train} pos / {len(y_train)} train, {n_pos_val} pos / {len(y_val)} val", flush=True)

    model = regextree.RegexBoostClassifier(
        pool_size=pool_size,
        step_count=step_count,
        eta=eta,
        llm="gemini",
        sample_size=20,
        verbose=True,
    )

    t0 = time.time()
    model.fit(x_train, y_train.tolist())
    elapsed = time.time() - t0

    y_pred = model.predict(x_val)
    acc = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, zero_division=0)
    prec = precision_score(y_val, y_pred, zero_division=0)
    rec = recall_score(y_val, y_pred, zero_division=0)

    print(f"  acc={acc:.4f} f1={f1:.4f} prec={prec:.4f} rec={rec:.4f} ({elapsed:.1f}s)", flush=True)
    for rule in model.rules():
        print(f"    {rule.id}: r'{rule.regex}' -> class {rule.target_class} (w={rule.weight:.4f})", flush=True)

    return {
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
        results.append(evaluate_cluster(df, cid))

    results_df = pd.DataFrame(results)
    print("\n\nSummary:", flush=True)
    print(results_df.to_string(index=False), flush=True)
    print(f"\nMean F1: {results_df['f1'].mean():.4f}", flush=True)
    print(f"Mean Accuracy: {results_df['accuracy'].mean():.4f}", flush=True)
