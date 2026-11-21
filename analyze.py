"""Rebuild the OASIS poster comparison and evaluate baseline visits separately."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, StackingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

SEED = 42
MRI = ["eTIV", "nWBV"]
CLINICAL = MRI + ["MMSE"]
RANK_ONE = CLINICAL + ["CDR"]


def pipeline(estimator):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), estimator)


def models():
    lr = LogisticRegression(max_iter=1000)
    forest = RandomForestClassifier(n_estimators=200, random_state=SEED, n_jobs=1)
    svm = SVC()
    stacking = StackingClassifier(
        estimators=[("lr", pipeline(lr)), ("forest", pipeline(forest)), ("svm", pipeline(svm))],
        final_estimator=LogisticRegression(max_iter=1000),
        cv=StratifiedKFold(5, shuffle=True, random_state=SEED))
    return {
        "Majority baseline": DummyClassifier(strategy="most_frequent"),
        "Logistic regression": pipeline(lr),
        "KNN": pipeline(KNeighborsClassifier()),
        "Gaussian naive Bayes": pipeline(GaussianNB()),
        "Decision tree": pipeline(DecisionTreeClassifier(random_state=SEED)),
        "SVM": pipeline(svm),
        "Stacking": stacking,
        "Random forest": pipeline(forest),
        "AdaBoost": pipeline(AdaBoostClassifier(random_state=SEED)),
        "XGBoost": pipeline(XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1,
                                          tree_method="hist", n_jobs=1, random_state=SEED))
    }


def scores(name, protocol, y, prediction):
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return dict(protocol=protocol, model=name, n=len(y), tn=tn, fp=fp, fn=fn, tp=tp,
                accuracy=accuracy_score(y, prediction),
                balanced_accuracy=balanced_accuracy_score(y, prediction),
                f1_positive=f1_score(y, prediction, zero_division=0),
                f1_macro=f1_score(y, prediction, average="macro", zero_division=0))


def analyze(data):
    if data["MRI ID"].duplicated().any() or data[["Subject ID", "Group", "CDR", "Visit"]].isna().any().any():
        raise ValueError("Expected unique session IDs and complete IDs, groups, visits and CDR.")
    if set(data.Group) != {"Demented", "Nondemented", "Converted"}:
        raise ValueError("Unexpected diagnostic group labels.")
    data = data.sort_values(["Subject ID", "Visit"]).reset_index(drop=True)
    rows = []

    # Poster-informed sensitivity analysis. Converted is a subject-level label.
    y = data.Group.ne("Nondemented").astype(int)
    train, test = train_test_split(np.arange(len(y)), test_size=.3, random_state=SEED, stratify=y)
    overlap = len(set(data.loc[train, "Subject ID"]) & set(data.loc[test, "Subject ID"]))
    for protocol, features in [("visit_split_with_cdr", RANK_ONE), ("visit_split_without_cdr", CLINICAL)]:
        for name, model in models().items():
            model.fit(data.loc[train, features], y.iloc[train])
            rows.append(scores(name, protocol, y.iloc[test], model.predict(data.loc[test, features])))

    # Reanalysis: one first visit per person. The target is that visit's CDR > 0.
    baseline = data.loc[data.Visit.eq(1)].reset_index(drop=True)
    if len(baseline) != data["Subject ID"].nunique() or baseline["Subject ID"].duplicated().any():
        raise ValueError("Expected exactly one baseline record for every participant.")
    y = baseline.CDR.gt(0).astype(int)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(baseline, y))
    for protocol, features in [("baseline_mri_mmse", CLINICAL), ("baseline_mri_only", MRI),
                               ("baseline_mmse_only", ["MMSE"])]:
        for name, model in models().items():
            predictions = cross_val_predict(model, baseline[features], y, cv=folds)
            rows.append(scores(name, protocol, y, predictions))
    print(f"{len(data)} visits, {len(baseline)} participants; visit split {len(train)}/{len(test)}; "
          f"{overlap} participants in both visit partitions; {y.sum()} positive baseline visits.")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Local oasis_longitudinal.csv")
    parser.add_argument("--output", type=Path, default=Path("results/reanalysis.csv"))
    args = parser.parse_args()
    results = analyze(pd.read_csv(args.csv))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False, float_format="%.6f")
