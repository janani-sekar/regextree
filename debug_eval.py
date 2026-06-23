"""Minimal debug test to find where the hang is."""

import os
import sys

os.environ["GEMINI_API_KEY"] = "AIzaSyB_fiIiYzDLsfjCCR0hUj97GRdux_G2Q4o"

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, "/Users/jsekar/repos/regextree/src")

print("loading data...", flush=True)
df = pd.read_feather("/Users/jsekar/Downloads/step_cluster.feather")
print(f"loaded {df.shape}", flush=True)

cluster_id = 4
y_binary = (df["cluster_idx"] == cluster_id).astype(int).values
x_all = df["step_prev_text"].tolist()

x_train, x_val, y_train, y_val = train_test_split(
    x_all, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)
print(f"split: {sum(y_train)} pos / {len(y_train)} train", flush=True)

# Test scoring manually
from regextree.scoring import softmax, compute_gradients, find_hardest_class
labels = np.array(y_train)
n_classes = 2
counts = np.bincount(labels, minlength=n_classes).astype(float)
priors = counts / counts.sum()
logits = np.log(priors + 1e-12)[None, :].repeat(len(x_train), axis=0)
print(f"logits shape: {logits.shape}, priors: {priors}", flush=True)

grads = compute_gradients(logits, labels)
print(f"gradients computed, shape: {grads.shape}", flush=True)

target = find_hardest_class(grads, labels)
print(f"hardest class: {target}", flush=True)

# Test sampling
from regextree.boost import RegexBoostClassifier
model = RegexBoostClassifier(pool_size=5, step_count=1, eta=0.3, llm="gemini", sample_size=15, verbose=True)
model._n_classes = n_classes
model._classes = np.array([0, 1])
positives, negatives = model._sample_examples(x_train, labels, grads, target)
print(f"sampled {len(positives)} pos, {len(negatives)} neg", flush=True)
print(f"pos examples: {positives[:3]}", flush=True)
print(f"neg examples: {negatives[:3]}", flush=True)

# Test LLM call
from regextree.prompt import generate_candidates
print("calling LLM...", flush=True)
candidates = generate_candidates(model.client, int(model._classes[target]), positives, negatives, 5)
print(f"got {len(candidates)} candidates: {candidates}", flush=True)

# Test matching
import re
print("evaluating candidates against training data...", flush=True)
for i, regex in enumerate(candidates):
    compiled = re.compile(regex)
    matches = np.array([compiled.search(t) is not None for t in x_train], dtype=bool)
    print(f"  candidate {i}: r'{regex}' matches {matches.sum()}/{len(x_train)}", flush=True)

print("ALL DONE", flush=True)
