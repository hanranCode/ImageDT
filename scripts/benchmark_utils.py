"""Reproduce the ROC threshold scan comparison against the legacy algorithm."""

import json
from time import perf_counter

import numpy as np


def benchmark(samples=5000):
    rng = np.random.default_rng(42)
    probabilities = rng.random(samples)
    matches = rng.integers(0, 2, samples)
    thresholds = np.arange(0, 1.001, 0.001)
    start = perf_counter()
    expected = []
    for threshold in thresholds:
        tp = fp = 0
        for probability, match in zip(probabilities, matches, strict=True):
            if probability >= threshold:
                if match:
                    tp += 1
                else:
                    fp += 1
        expected.append(
            (tp / max(int(matches.sum()), 1), fp / max(int((matches == 0).sum()), 1))
        )
    legacy_seconds = perf_counter() - start
    start = perf_counter()
    positive = np.sort(probabilities[matches == 1])
    negative = np.sort(probabilities[matches == 0])
    actual = np.column_stack(
        (
            (len(positive) - np.searchsorted(positive, thresholds))
            / max(len(positive), 1),
            (len(negative) - np.searchsorted(negative, thresholds))
            / max(len(negative), 1),
        )
    )
    new_seconds = perf_counter() - start
    np.testing.assert_allclose(actual, expected)
    return {
        "samples": samples,
        "thresholds": len(thresholds),
        "legacy_seconds": legacy_seconds,
        "new_seconds": new_seconds,
        "speedup": legacy_seconds / new_seconds,
        "numerical_equivalence": True,
    }


if __name__ == "__main__":
    print(json.dumps(benchmark(), indent=2))
