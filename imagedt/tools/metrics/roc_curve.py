import os

import matplotlib.pyplot as plt
import numpy as np

import imagedt


def roc(test_file, show_fig=False):
    imagedt.file.readlines(test_file)

    import csv

    with open(test_file, encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.reader(stream))
    if not rows:
        raise ValueError("ROC input is empty")
    try:
        int(rows[0][-1])
    except ValueError:
        rows = rows[1:]
    if not rows:
        raise ValueError("ROC input has no samples")
    probability = np.asarray([row[-2] for row in rows], dtype=float)
    matches = np.asarray([row[-1] for row in rows], dtype=int)
    if (
        np.any(~np.isfinite(probability))
        or np.any((probability < 0) | (probability > 1))
        or np.any((matches != 0) & (matches != 1))
    ):
        raise ValueError("expected finite probabilities in [0, 1] and binary matches")
    thresholds = np.arange(0, 1.001, 0.001)
    positive = np.sort(probability[matches == 1])
    negative = np.sort(probability[matches == 0])
    # O(N log N + T log N), without a threshold-by-sample Python loop.
    tpr = (len(positive) - np.searchsorted(positive, thresholds, side="left")) / max(
        len(positive), 1
    )
    fnr = (len(negative) - np.searchsorted(negative, thresholds, side="left")) / max(
        len(negative), 1
    )
    thresvalue = thresholds
    save_lines = np.column_stack((fnr, tpr, thresholds)).tolist()

    save_dir = os.path.dirname(test_file)

    imagedt.file.write_csv(save_lines, os.path.join(save_dir, "thres-fpr-tpr.csv"))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xscale("log")
    plt.grid()
    plt.xlabel("FPR")
    plt.ylabel("TPR/THRES")
    plt.title("ROC-THRES")
    ax.plot(fnr, tpr, color="red")
    ax.plot(fnr, thresvalue, color="green")
    plt.legend(("roc", "fpr-thres"), loc="best")

    plt.savefig(os.path.join(save_dir, "roc.png"), dpi=300)
    print(f"ROC curve figure save: {save_dir}")
    if show_fig:
        plt.show()
    plt.close(fig)
