"""Reconstruct the poster's exploratory analysis and nine-model comparison."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from boruta import BorutaPy
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from analyze import RANK_ONE, SEED, models, poster_split, scores

# Eight candidate variables and ranks transcribed from the poster's table.
POSTER_RANK = {"M/F": 5, "Age": 3, "EDUC": 4, "MMSE": 1, "CDR": 1,
               "eTIV": 1, "nWBV": 1, "ASF": 2}


def reconstruct(csv, output):
    data, y, train, test = poster_split(pd.read_csv(csv))
    features = data.loc[train, list(POSTER_RANK)].copy()
    features["M/F"] = features["M/F"].map({"M": 1, "F": 0})
    imputed = SimpleImputer(strategy="median").fit_transform(features)
    standardized = StandardScaler().fit_transform(imputed)
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(standardized, columns=features.columns).corr().to_csv(
        output / "feature-correlations.csv", float_format="%.6f")

    pca = PCA(svd_solver="full").fit(standardized)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    pcs = pd.DataFrame({"component": np.arange(1, len(cumulative) + 1),
                        "variance_fraction": pca.explained_variance_ratio_,
                        "cumulative_fraction": cumulative})
    pcs.to_csv(output / "pca.csv", index=False, float_format="%.6f")

    forest = RandomForestClassifier(max_depth=5, n_jobs=1, random_state=SEED)
    selector = BorutaPy(forest, n_estimators="auto", max_iter=100, random_state=SEED)
    selector.fit(standardized, y.iloc[train].to_numpy())
    ranks = pd.DataFrame({"feature": features.columns, "poster_rank": list(POSTER_RANK.values()),
                          "reconstructed_rank": selector.ranking_, "confirmed": selector.support_,
                          "tentative": selector.support_weak_})
    ranks.to_csv(output / "boruta.csv", index=False)

    # Keep the displayed rank-one set fixed so the new ranks can be compared
    # independently and scores match the archive's visit-level reconstruction.
    results = []
    for name, model in models().items():
        if name == "Majority baseline":
            continue
        model.fit(data.loc[train, RANK_ONE], y.iloc[train])
        for partition, indices in (("train", train), ("test", test)):
            prediction = model.predict(data.loc[indices, RANK_ONE])
            results.append(scores(name, partition, y.iloc[indices], prediction))
    metrics = pd.DataFrame(results).rename(columns={"protocol": "partition"})
    metrics.to_csv(output / "models.csv", index=False, float_format="%.6f")
    plot_summary(pcs, metrics, output / "poster-reconstruction.svg")
    components = int(np.searchsorted(cumulative, 0.95) + 1)
    print(f"Training visits: {len(train)}. Test visits: {len(test)}. "
          f"Components for 95% variance: {components}.")
    print("Confirmed Boruta features:", ", ".join(ranks.loc[ranks.confirmed, "feature"]))


def plot_summary(pcs, metrics, path):
    plt.rcParams.update({"svg.fonttype": "none", "svg.hashsalt": "oasis-poster"})
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), layout="constrained")
    ax, bars = axes
    ax.plot(pcs.component, pcs.cumulative_fraction * 100, marker="o", color="#155e75")
    ax.axhline(95, color="#945019", ls="--", label="95% variance")
    ax.set(xlabel="Number of components", ylabel="Cumulative variance (%)",
           title="PCA on 261 training visits", ylim=(0, 105), xticks=pcs.component)
    ax.legend(loc="lower right", frameon=False)
    held = metrics.loc[metrics.partition.eq("test")].sort_values("accuracy")
    bars.barh(held.model, held.accuracy * 100, color="#397768")
    bars.set(xlim=(0, 106), xlabel="Accuracy (%)", title="Nine models on 112 test visits")
    for i, accuracy in enumerate(held.accuracy):
        bars.text(accuracy * 100 + 1, i, f"{accuracy:.1%}", va="center", fontsize=8)
    for panel in axes:
        panel.spines[["top", "right"]].set_visible(False)
    fig.savefig(path, metadata={"Date": None})
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/poster-reconstruction"))
    args = parser.parse_args()
    reconstruct(args.csv, args.output)
