from __future__ import annotations

import re
from typing import Sequence

from .llm import LLMClient

REGEX_GENERATION_PROMPT = """\
You are helping build an interpretable text classifier. Your job is to propose \
Python regex patterns that match one group of text strings but not another.

TARGET CLASS: {target_class}

POSITIVE EXAMPLES (should match):
{positive_examples}

NEGATIVE EXAMPLES (should NOT match):
{negative_examples}

Generate exactly {pool_size} distinct Python regex patterns. Each pattern should:
- Be a valid Python regex (re module compatible)
- Match as many positive examples as possible
- Match as few negative examples as possible
- Be human-readable and interpretable (prefer simple character classes over complex lookaheads)

Return ONLY the patterns, one per line, with no additional text or explanation.
Each line should contain exactly one regex pattern (no quotes, no numbering).
"""

DESCRIPTION_PROMPT = """\
Given this Python regex pattern used for text classification:

Pattern: {regex}
Target class: {target_class}

Example matches:
{example_matches}

Write a short (1 sentence) human-readable description of what this pattern detects. \
Be specific about the textual feature it captures. Return only the description, nothing else.
"""


def build_generation_prompt(
    target_class: int,
    positives: Sequence[str],
    negatives: Sequence[str],
    pool_size: int,
    max_examples: int = 15,
) -> str:
    pos_display = "\n".join(f"  {t}" for t in positives[:max_examples])
    neg_display = "\n".join(f"  {t}" for t in negatives[:max_examples])
    return REGEX_GENERATION_PROMPT.format(
        target_class=target_class,
        positive_examples=pos_display,
        negative_examples=neg_display,
        pool_size=pool_size,
    )


def build_description_prompt(
    regex: str,
    target_class: int,
    example_matches: Sequence[str],
    max_examples: int = 10,
) -> str:
    examples_display = "\n".join(f"  {t}" for t in example_matches[:max_examples])
    return DESCRIPTION_PROMPT.format(
        regex=regex,
        target_class=target_class,
        example_matches=examples_display,
    )


def parse_regex_candidates(response: str) -> list[str]:
    """Extract valid regex patterns from LLM response text."""
    candidates = []
    for line in response.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        # Strip leading numbering like "1. " or "1) "
        line = re.sub(r"^\d+[\.\)]\s*", "", line)
        # Strip surrounding backticks or quotes
        line = line.strip("`\"'")
        if not line:
            continue
        # Validate that it compiles
        try:
            re.compile(line)
            candidates.append(line)
        except re.error:
            continue
    return candidates


def generate_candidates(
    client: LLMClient,
    target_class: int,
    positives: Sequence[str],
    negatives: Sequence[str],
    pool_size: int,
    max_retries: int = 2,
) -> list[str]:
    """Prompt the LLM and return up to pool_size valid regex candidates."""
    prompt = build_generation_prompt(target_class, positives, negatives, pool_size)

    all_candidates: list[str] = []
    for _ in range(1 + max_retries):
        response = client.generate(prompt)
        parsed = parse_regex_candidates(response)
        all_candidates.extend(parsed)
        if len(all_candidates) >= pool_size:
            break

    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for c in all_candidates:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique[:pool_size]


def generate_description(
    client: LLMClient,
    regex: str,
    target_class: int,
    texts: Sequence[str],
) -> str:
    """Generate a human-readable description for a regex rule."""
    compiled = re.compile(regex)
    matches = [t for t in texts if compiled.search(t)]
    if not matches:
        return f"Pattern for class {target_class} (no matches in sample)"
    prompt = build_description_prompt(regex, target_class, matches)
    return client.generate(prompt).strip()
