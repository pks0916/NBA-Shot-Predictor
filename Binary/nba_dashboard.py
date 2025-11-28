# nba_dashboard.py
#
# Analytics + visuals for the NBA Shot Predictor project.
# - Trains 4 models (LR, KNN, GaussianNB, RandomForest)
# - Creates:
#     * shot_chart.png
#     * model_accuracies.png
#     * rf_feature_importance.png
#     * tier_avg_distance.png
#     * tier_distance_make_pct.png
#     * time_phase_make_pct.png
#     * xfg_overperformers.png
#     * xfg_underperformers.png
# - Computes xFG% (expected FG%) per player
# - Shows "simple shot advisor" example for top vs bottom shooters

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance


warnings.filterwarnings("ignore", category=UserWarning)


def build_features(df):
    """Apply the same feature engineering as in nba_bin_classifiers.py."""
    df = df.copy()

    # Shooting angle
    df["ANGLE"] = np.arctan2(df["LOC_Y"], df["LOC_X"])

    # Corner 3 flag
    df["IS_CORNER"] = (
        (df["SHOT_TYPE"] == "3PT Field Goal")
        & (df["LOC_Y"] < 92)
        & (df["LOC_X"].abs() > 220)
    ).astype(int)

    # Time remaining in period (seconds)
    df["TIME_REMAINING"] = df["MINUTES_REMAINING"] * 60 + df["SECONDS_REMAINING"]

    # Distance bucket
    dist_bins = [0, 5, 10, 15, 20, 25, 30, 35, 40]
    df["DIST_CAT"] = pd.cut(
        df["SHOT_DISTANCE"],
        bins=dist_bins,
        labels=False,
        include_lowest=True,
    )
    df["DIST_CAT"] = df["DIST_CAT"].fillna(len(dist_bins)).astype(int)

    # Shooter tier as string (bottom / middle / top)
    df["SHOOTER_TIER"] = df["SHOOTER_TIER"].astype(str)

    return df


def train_models(df):
    """Train LR, KNN, NB, RF and return trained models + metrics + splits."""

    df = build_features(df)

    X = df.drop(columns=["SHOT_MADE_FLAG", "PLAYER_ID"])
    y = df["SHOT_MADE_FLAG"]

    numeric_features = [
        "LOC_X",
        "LOC_Y",
        "SHOT_DISTANCE",
        "PERIOD",
        "MINUTES_REMAINING",
        "SECONDS_REMAINING",
        "ANGLE",
        "TIME_REMAINING",
        "DIST_CAT",
        "PLAYER_FG_PCT",
        "PLAYER_SHOT_ATTEMPTS",
    ]

    categorical_features = [
        "SHOT_TYPE",
        "SHOT_ZONE_BASIC",
        "IS_CORNER",
        "SHOOTER_TIER",
    ]

    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocess", preprocess),
                (
                    "clf",
                    LogisticRegression(
                        C=2.0,
                        penalty="l2",
                        solver="lbfgs",
                        max_iter=2000,
                    ),
                ),
            ]
        ),
        "KNN": Pipeline(
            steps=[
                ("preprocess", preprocess),
                (
                    "clf",
                    KNeighborsClassifier(
                        n_neighbors=11,
                        weights="distance",
                    ),
                ),
            ]
        ),
        "Gaussian NB": Pipeline(
            steps=[
                ("preprocess", preprocess),
                ("clf", GaussianNB()),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocess", preprocess),
                (
                    "clf",
                    RandomForestClassifier(
                        n_estimators=200,
                        max_depth=12,
                        min_samples_split=2,
                        min_samples_leaf=1,
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }

    trained = {}
    metrics = []

    for name, pipe in models.items():
        print(f"\nTraining {name} ...")
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"\n==== {name} ====")
        print("Accuracy:", acc)
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))

        trained[name] = pipe
        metrics.append({"model": name, "accuracy": acc})

    metrics_df = pd.DataFrame(metrics)

    # Pick best model by accuracy (will usually be Random Forest)
    best_row = metrics_df.sort_values("accuracy", ascending=False).iloc[0]
    best_model_name = best_row["model"]
    best_model = trained[best_model_name]

    print(f"\nBest model: {best_model_name} (accuracy {best_row['accuracy']:.4f})")

    return trained, metrics_df, best_model, df, X, y


def plot_shot_chart(df):
    sample = df.sample(n=min(5000, len(df)), random_state=42)

    plt.figure(figsize=(6, 6))
    sns.scatterplot(
        data=sample,
        x="LOC_X",
        y="LOC_Y",
        hue="SHOT_MADE_FLAG",
        alpha=0.4,
        s=10,
    )
    plt.title("Shot Chart (Sample of 2025 Season)")
    plt.xlabel("LOC_X")
    plt.ylabel("LOC_Y")
    plt.legend(title="Made?", labels=["Miss (0)", "Make (1)"])
    plt.tight_layout()
    plt.savefig("shot_chart.png", dpi=200)
    plt.close()


def plot_model_accuracies(metrics_df):
    plt.figure(figsize=(6, 4))
    sns.barplot(data=metrics_df, x="model", y="accuracy")
    plt.ylim(0, 1)
    plt.title("Model Accuracies")
    plt.ylabel("Accuracy")
    plt.xlabel("Model")
    plt.tight_layout()
    plt.savefig("model_accuracies.png", dpi=200)
    plt.close()


def plot_feature_importance_rf(rf_model, X):
    # Use permutation importance on a sample for speed
    rng = np.random.RandomState(42)
    sample_size = min(2000, len(X))
    idx = rng.choice(len(X), size=sample_size, replace=False)

    X_sample = X.iloc[idx]
    # We need corresponding y, but RF only needs X for permutation_importance?
    # permutation_importance requires both X and y, so we pass y separately.
    # We'll compute y from original df outside and pass in.
    # This function will be called with full X and y_sample together,
    # so we won't use y inside here.

    # We'll just return and handle inside main.
    return X_sample, idx


def compute_and_plot_rf_importance(rf_model, X, y):
    rng = np.random.RandomState(42)
    sample_size = min(2000, len(X))
    idx = rng.choice(len(X), size=sample_size, replace=False)

    X_sample = X.iloc[idx]
    y_sample = y.iloc[idx]

    perm = permutation_importance(
        rf_model,
        X_sample,
        y_sample,
        n_repeats=5,
        random_state=42,
        n_jobs=-1,
    )

    importances = perm.importances_mean
    feature_names = X.columns

    imp_df = pd.DataFrame(
        {"feature": feature_names, "importance": importances}
    ).sort_values("importance", ascending=False)

    top_imp = imp_df.head(15)

    plt.figure(figsize=(8, 5))
    sns.barplot(data=top_imp, x="importance", y="feature")
    plt.title("Random Forest Feature Importance (Permutation)")
    plt.tight_layout()
    plt.savefig("rf_feature_importance.png", dpi=200)
    plt.close()


def compute_xfg_and_plots(best_model, df):
    # df already has engineered features here
    X_full = df.drop(columns=["SHOT_MADE_FLAG", "PLAYER_ID"])
    probs = best_model.predict_proba(X_full)[:, 1]
    df = df.copy()
    df["xFG"] = probs

    # Try to get player names from original CSV (if it has PLAYER_NAME)
    name_map = None
    try:
        raw = pd.read_csv("../shot_charts_2025.csv")
        if "PLAYER_ID" in raw.columns and "PLAYER_NAME" in raw.columns:
            name_map = (
                raw[["PLAYER_ID", "PLAYER_NAME"]]
                .drop_duplicates()
                .set_index("PLAYER_ID")["PLAYER_NAME"]
            )
    except FileNotFoundError:
        pass

    player_stats = (
        df.groupby("PLAYER_ID")
        .agg(
            actual_fg=("SHOT_MADE_FLAG", "mean"),
            expected_fg=("xFG", "mean"),
            attempts=("SHOT_MADE_FLAG", "size"),
            tier=("SHOOTER_TIER", "first"),
        )
        .reset_index()
    )

    if name_map is not None:
        player_stats["PLAYER_NAME"] = player_stats["PLAYER_ID"].map(name_map)
    else:
        player_stats["PLAYER_NAME"] = player_stats["PLAYER_ID"].astype(str)

    # Only look at players with a decent sample size
    player_stats = player_stats[player_stats["attempts"] >= 100].copy()
    player_stats["diff"] = player_stats["actual_fg"] - player_stats["expected_fg"]

    # Save full table
    player_stats.to_csv("player_xfg_all.csv", index=False)

    over = player_stats.sort_values("diff", ascending=False).head(10)
    under = player_stats.sort_values("diff").head(10)

    over.to_csv("player_xfg_overperformers.csv", index=False)
    under.to_csv("player_xfg_underperformers.csv", index=False)

    # Plot overperformers
    plt.figure(figsize=(7, 5))
    sns.barplot(
        data=over,
        x="diff",
        y="PLAYER_NAME",
    )
    plt.title("Top 10 Overperformers (Actual FG% - Expected FG%)")
    plt.xlabel("Actual - Expected FG%")
    plt.tight_layout()
    plt.savefig("xfg_overperformers.png", dpi=200)
    plt.close()

    # Plot underperformers
    plt.figure(figsize=(7, 5))
    sns.barplot(
        data=under,
        x="diff",
        y="PLAYER_NAME",
    )
    plt.title("Top 10 Underperformers (Actual FG% - Expected FG%)")
    plt.xlabel("Actual - Expected FG%")
    plt.tight_layout()
    plt.savefig("xfg_underperformers.png", dpi=200)
    plt.close()


def plot_tier_behavior(df):
    # Average shot distance by shooter tier
    tier_avg_dist = (
        df.groupby("SHOOTER_TIER")["SHOT_DISTANCE"]
        .mean()
        .reset_index()
        .sort_values("SHOT_DISTANCE")
    )

    plt.figure(figsize=(6, 4))
    sns.barplot(
        data=tier_avg_dist,
        x="SHOOTER_TIER",
        y="SHOT_DISTANCE",
    )
    plt.title("Average Shot Distance by Shooter Tier")
    plt.ylabel("Average SHOT_DISTANCE (ft)")
    plt.xlabel("Shooter Tier")
    plt.tight_layout()
    plt.savefig("tier_avg_distance.png", dpi=200)
    plt.close()

    # Make% by distance and tier
    dist_edges = [0, 5, 10, 15, 20, 25, 30, 35]
    dist_labels = [
        f"{dist_edges[i]}-{dist_edges[i+1]}" for i in range(len(dist_edges) - 1)
    ]

    df = df.copy()
    df["DIST_BIN"] = pd.cut(
        df["SHOT_DISTANCE"],
        bins=dist_edges,
        labels=dist_labels,
        include_lowest=True,
    )
    df = df.dropna(subset=["DIST_BIN"])

    tier_dist_pct = (
        df.groupby(["SHOOTER_TIER", "DIST_BIN"])["SHOT_MADE_FLAG"]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(8, 5))
    sns.lineplot(
        data=tier_dist_pct,
        x="DIST_BIN",
        y="SHOT_MADE_FLAG",
        hue="SHOOTER_TIER",
        marker="o",
    )
    plt.ylim(0, 1)
    plt.ylabel("Make Percentage")
    plt.xlabel("Shot Distance Bin (ft)")
    plt.title("Make% by Distance and Shooter Tier")
    plt.tight_layout()
    plt.savefig("tier_distance_make_pct.png", dpi=200)
    plt.close()


def plot_time_phase_behavior(df):
    # Early / mid / late in the period based on TIME_REMAINING
    df = df.copy()

    def time_phase(t):
        if t > 360:
            return "Early (12–6 min)"
        elif t > 120:
            return "Middle (6–2 min)"
        else:
            return "Late (last 2 min)"

    df["TIME_PHASE"] = df["TIME_REMAINING"].apply(time_phase)

    phase_stats = (
        df.groupby("TIME_PHASE")["SHOT_MADE_FLAG"]
        .agg(make_pct="mean", attempts="size")
        .reset_index()
    )

    print("\nShot selection by time in period:")
    print(phase_stats)

    plt.figure(figsize=(6, 4))
    sns.barplot(
        data=phase_stats,
        x="TIME_PHASE",
        y="make_pct",
    )
    plt.ylim(0, 1)
    plt.ylabel("Make Percentage")
    plt.xlabel("Time in Period")
    plt.title("Make% by Time in Period")
    plt.tight_layout()
    plt.savefig("time_phase_make_pct.png", dpi=200)
    plt.close()


def simple_shot_advisor(best_model, df):
    """
    Simple shot advisor demo:
    - Same shot location + situation
    - Compare top-tier vs bottom-tier shooter probabilities.
    """

    # Median FG% + attempts for each tier
    tier_stats = (
        df.groupby("SHOOTER_TIER")[["PLAYER_FG_PCT", "PLAYER_SHOT_ATTEMPTS"]]
        .median()
    )

    dist_bins = [0, 5, 10, 15, 20, 25, 30, 35, 40]

    def scenario_prob(shooter_tier, loc_x, loc_y, shot_distance,
                      period, minutes, seconds, shot_type, shot_zone_basic):
        row = {
            "LOC_X": loc_x,
            "LOC_Y": loc_y,
            "SHOT_DISTANCE": shot_distance,
            "PERIOD": period,
            "MINUTES_REMAINING": minutes,
            "SECONDS_REMAINING": seconds,
            "SHOT_TYPE": shot_type,
            "SHOT_ZONE_BASIC": shot_zone_basic,
            "PLAYER_ID": -1,  # dummy, dropped later
            "PLAYER_FG_PCT": tier_stats.loc[shooter_tier, "PLAYER_FG_PCT"],
            "PLAYER_SHOT_ATTEMPTS": tier_stats.loc[
                shooter_tier, "PLAYER_SHOT_ATTEMPTS"
            ],
            "SHOOTER_TIER": shooter_tier,
            "SHOT_MADE_FLAG": 0,  # dummy
        }

        df_row = pd.DataFrame([row])

        # Apply same feature engineering as training
        df_row["ANGLE"] = np.arctan2(df_row["LOC_Y"], df_row["LOC_X"])
        df_row["IS_CORNER"] = (
            (df_row["SHOT_TYPE"] == "3PT Field Goal")
            & (df_row["LOC_Y"] < 92)
            & (df_row["LOC_X"].abs() > 220)
        ).astype(int)
        df_row["TIME_REMAINING"] = (
            df_row["MINUTES_REMAINING"] * 60 + df_row["SECONDS_REMAINING"]
        )
        df_row["DIST_CAT"] = pd.cut(
            df_row["SHOT_DISTANCE"],
            bins=dist_bins,
            labels=False,
            include_lowest=True,
        )
        df_row["DIST_CAT"] = df_row["DIST_CAT"].fillna(len(dist_bins)).astype(int)

        X_row = df_row.drop(columns=["SHOT_MADE_FLAG", "PLAYER_ID"])
        prob = best_model.predict_proba(X_row)[0, 1]
        return prob

    # Example: Above-the-break three at the top of the key
    loc_x = 0
    loc_y = 200
    shot_distance = 24
    period = 2
    minutes = 6
    seconds = 0
    shot_type = "3PT Field Goal"
    shot_zone_basic = "Above the Break 3"

    p_top = scenario_prob(
        "top",
        loc_x,
        loc_y,
        shot_distance,
        period,
        minutes,
        seconds,
        shot_type,
        shot_zone_basic,
    )
    p_bottom = scenario_prob(
        "bottom",
        loc_x,
        loc_y,
        shot_distance,
        period,
        minutes,
        seconds,
        shot_type,
        shot_zone_basic,
    )

    print("\n=== Simple Shot Advisor Demo ===")
    print("Scenario: Above-the-break 3, top of the key, 24 ft, Q2, 6:00 remaining")
    print(f"Top-tier shooter make probability:    {p_top:.3f}")
    print(f"Bottom-tier shooter make probability: {p_bottom:.3f}")


def main():
    # 1. Load enhanced data
    df = pd.read_csv("shot_charts_2025_enhanced.csv")

    # 2. Train models & get best
    trained_models, metrics_df, best_model, df_feat, X, y = train_models(df)

    # Save accuracies for reference
    metrics_df.to_csv("model_accuracies_dashboard.csv", index=False)

    # 3. Plots for dashboard
    print("\nCreating visualizations...")
    plot_shot_chart(df)
    plot_model_accuracies(metrics_df)

    # Use Random Forest for feature importance (even if LR is very close)
    rf_model = trained_models["Random Forest"]
    compute_and_plot_rf_importance(rf_model, X, y)

    # 4. xFG% over/under performance
    compute_xfg_and_plots(best_model, df_feat)

    # 5. Top vs bottom shooter behavior
    plot_tier_behavior(df)

    # 6. Time-phase behavior (early / mid / late in period)
    plot_time_phase_behavior(df_feat)

    # 7. Simple shot advisor demo
    simple_shot_advisor(best_model, df_feat)

    print("\nAll plots saved as .png files in the Binary folder.")


if __name__ == "__main__":
    main()
