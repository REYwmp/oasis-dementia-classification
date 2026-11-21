"""Compare visit and participant splits, then count paired baseline prediction changes."""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold, cross_val_predict

from analyze import CLINICAL, SEED, baseline_folds, models, poster_split, scores


def compare_splits(data, target):
    groups = data["Subject ID"]
    features = data[CLINICAL]
    model = models()["XGBoost"]
    protocols = {
        "by_visit": StratifiedKFold(5, shuffle=True, random_state=SEED).split(features, target),
        "by_participant": StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(
            features, target, groups
        ),
    }
    summary, audits = [], []
    for name, partitions in protocols.items():
        folds = list(partitions)
        for number, (train, test) in enumerate(folds, start=1):
            seen = groups.iloc[test].isin(groups.iloc[train])
            if name == "by_participant" and seen.any():
                raise ValueError("A held-out participant appeared in training.")
            audits.append(dict(protocol=name, fold=number, train_visits=len(train),
                               test_visits=len(test), positive_test_visits=int(target.iloc[test].sum()),
                               test_people=groups.iloc[test].nunique(),
                               test_visits_from_people_seen_in_training=int(seen.sum())))
        predictions = cross_val_predict(model, features, target, cv=folds)
        summary.append(scores("XGBoost", name, target, predictions))
    return pd.DataFrame(summary), pd.DataFrame(audits)


def baseline_changes(data):
    baseline, target, folds = baseline_folds(data)
    model = models()["XGBoost"]
    cognitive = cross_val_predict(model, baseline[["MMSE"]], target, cv=folds)
    combined = cross_val_predict(model, baseline[CLINICAL], target, cv=folds)
    # Keep the joint counts for each recorded rating so both confusion matrices
    # and the number of people who changed prediction can be reconstructed.
    rows = []
    for actual in (0, 1):
        for before in (0, 1):
            for after in (0, 1):
                count = ((target == actual) & (cognitive == before) & (combined == after)).sum()
                rows.append(dict(cdr_above_zero=actual, mmse_prediction=before,
                                 mmse_mri_prediction=after, participants=int(count)))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    data, target, _, _ = poster_split(pd.read_csv(args.csv))
    summary, folds = compare_splits(data, target)
    changes = baseline_changes(data)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "split-comparison.csv", index=False, float_format="%.6f")
    folds.to_csv(args.output_dir / "split-folds.csv", index=False)
    changes.to_csv(args.output_dir / "paired-baseline-errors.csv", index=False)
