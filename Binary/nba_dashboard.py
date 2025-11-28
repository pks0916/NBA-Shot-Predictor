import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_data(path="shot_charts_2025_enhanced.csv"):
    df = pd.read_csv(path)

    df["ANGLE"] = np.arctan2(df["LOC_Y"], df["LOC_X"])

    df["IS_CORNER"] = (
        (df["SHOT_TYPE"] == "3PT Field Goal")
        & (df["LOC_Y"] < 92)
        & (df["LOC_X"].abs() > 220)
    ).astype(int)

    df["TIME_REMAINING"] = df["MINUTES_REMAINING"] * 60 + df["SECONDS_REMAINING"]

    dist_bins = [0, 5, 10, 15, 20, 25, 30, 35, 40]
    df["DIST_CAT"] = pd.cut(
        df["SHOT_DISTANCE"],
        bins=dist_bins,
        labels=False,
        include_lowest=True,
    )
    df["DIST_CAT"] = df["DIST_CAT"].fillna(len(dist_bins)).astype(int)

    df["SHOOTER_TIER"] = df["SHOOTER_TIER"].astype(str)

    return df


def get_xy(df):
    X = df.drop(columns=["SHOT_MADE_FLAG", "PLAYER_ID"])
    y = df["SHOT_MADE_FLAG"]

    num_cols = [
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

    cat_cols = [
        "SHOT_TYPE",
        "SHOT_ZONE_BASIC",
        "IS_CORNER",
        "SHOOTER_TIER",
    ]

    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    return X, y, pre, num_cols, cat_cols


def build_models(pre):
    models = {}

    models["Logistic Regression"] = Pipeline(
        steps=[
            ("pre", pre),
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
    )

    models["KNN"] = Pipeline(
        steps=[
            ("pre", pre),
            (
                "clf",
                KNeighborsClassifier(
                    n_neighbors=11,
                    weights="distance",
                ),
            ),
        ]
    )

    models["Gaussian NB"] = Pipeline(
        steps=[
            ("pre", pre),
            ("clf", GaussianNB()),
        ]
    )

    models["Random Forest"] = Pipeline(
        steps=[
            ("pre", pre),
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
    )

    return models


def make_shot_chart(df):
    sample_df = df.sample(min(5000, len(df)), random_state=42)

    plt.figure(figsize=(6, 7))

    colors = {0: "orange", 1: "steelblue"}

    for val in [0, 1]:
        part = sample_df[sample_df["SHOT_MADE_FLAG"] == val]
        plt.scatter(
            part["LOC_X"],
            part["LOC_Y"],
            s=5,
            alpha=0.4,
            label=f"{'Miss' if val==0 else 'Make'} ({val})",
            c=colors[val],
        )

    plt.xlabel("LOC_X")
    plt.ylabel("LOC_Y")
    plt.title("Shot Chart (Sample of 2025 Season)")
    plt.legend(title="Made?")
    plt.tight_layout()
    plt.savefig("shot_chart.png", dpi=200)
    plt.close()


def model_accuracy_plots(models, X_train, X_test, y_train, y_test):
    rows = []

    for name, m in models.items():
        m.fit(X_train, y_train)
        y_pred = m.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        rows.append({"model": name, "accuracy": acc})
        print(name, "accuracy:", acc)

    acc_df = pd.DataFrame(rows)
    acc_df.to_csv("model_accuracies_dashboard.csv", index=False)

    plt.figure(figsize=(6, 5))
    plt.bar(acc_df["model"], acc_df["accuracy"])
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.xlabel("Model")
    plt.title("Model Accuracies")
    plt.tight_layout()
    plt.savefig("model_accuracies.png", dpi=200)
    plt.close()

    return acc_df


def rf_feature_importance_plot(rf_pipe, X_test, y_test, feature_names=None):
    rf_pipe.fit(X_test, y_test)

    result = permutation_importance(
        rf_pipe,
        X_test,
        y_test,
        n_repeats=5,
        random_state=42,
        n_jobs=-1,
    )

    imp = result.importances_mean

    pre_fitted = rf_pipe.named_steps["pre"]
    try:
        feat_names = pre_fitted.get_feature_names_out()
    except Exception:
        feat_names = np.array([f"f{i}" for i in range(len(imp))])

    if len(feat_names) != len(imp):
        m = min(len(feat_names), len(imp))
        feat_names = feat_names[:m]
        imp = imp[:m]

    imp_df = pd.DataFrame(
        {"feature": feat_names, "importance": imp}
    ).sort_values("importance", ascending=False)

    plt.figure(figsize=(8, 6))
    plt.barh(imp_df["feature"], imp_df["importance"])
    plt.gca().invert_yaxis()
    plt.xlabel("importance")
    plt.title("Random Forest Feature Importance (Permutation)")
    plt.tight_layout()
    plt.savefig("rf_feature_importance.png", dpi=200)
    plt.close()

    return imp_df


def build_player_xfg(df, rf_pipe, X_all, y_all):
    rf_pipe.fit(X_all, y_all)
    probs = rf_pipe.predict_proba(X_all)[:, 1]

    tmp = df.copy()
    tmp["xfg_prob"] = probs

    grp = tmp.groupby(["PLAYER_ID", "PLAYER_NAME"])

    player_summary = grp["SHOT_MADE_FLAG"].mean().to_frame("actual_fg")
    player_summary["expected_fg"] = grp["xfg_prob"].mean()
    player_summary["attempts"] = grp["SHOT_MADE_FLAG"].count()
    player_summary["tier"] = grp["SHOOTER_TIER"].first()
    player_summary["diff"] = player_summary["actual_fg"] - player_summary["expected_fg"]

    player_summary = player_summary.reset_index()

    player_summary.to_csv("player_xfg_all.csv", index=False)

    top_over = player_summary.sort_values("diff", ascending=False).head(10)
    top_under = player_summary.sort_values("diff", ascending=True).head(10)

    top_over.to_csv("player_xfg_overperformers.csv", index=False)
    top_under.to_csv("player_xfg_underperformers.csv", index=False)

    plt.figure(figsize=(8, 6))
    plt.barh(top_over["PLAYER_NAME"], top_over["diff"])
    plt.xlabel("Actual - Expected FG%")
    plt.title("Top 10 Overperformers (Actual FG% - Expected FG%)")
    plt.tight_layout()
    plt.savefig("xfg_overperformers.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.barh(top_under["PLAYER_NAME"], top_under["diff"])
    plt.xlabel("Actual - Expected FG%")
    plt.title("Top 10 Underperformers (Actual FG% - Expected FG%)")
    plt.tight_layout()
    plt.savefig("xfg_underperformers.png", dpi=200)
    plt.close()


def tier_distance_plots(df):
    tier_dist = df.groupby("SHOOTER_TIER")["SHOT_DISTANCE"].mean().reset_index()

    plt.figure(figsize=(6, 5))
    plt.bar(tier_dist["SHOOTER_TIER"], tier_dist["SHOT_DISTANCE"])
    plt.xlabel("Shooter Tier")
    plt.ylabel("Average SHOT_DISTANCE (ft)")
    plt.title("Average Shot Distance by Shooter Tier")
    plt.tight_layout()
    plt.savefig("tier_avg_distance.png", dpi=200)
    plt.close()

    bins = [0, 5, 10, 15, 20, 25, 30, 35]
    labels = ["0-5", "5-10", "10-15", "15-20", "20-25", "25-30", "30-35"]
    df["DIST_BIN"] = pd.cut(
        df["SHOT_DISTANCE"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    )

    grp = df.groupby(
        ["SHOOTER_TIER", "DIST_BIN"], observed=False
    )["SHOT_MADE_FLAG"].mean().reset_index()

    plt.figure(figsize=(8, 6))
    for tier in ["bottom", "middle", "top"]:
        part = grp[grp["SHOOTER_TIER"] == tier]
        plt.plot(part["DIST_BIN"], part["SHOT_MADE_FLAG"], marker="o", label=tier)

    plt.ylim(0, 1)
    plt.xlabel("Shot Distance Bin (ft)")
    plt.ylabel("Make Percentage")
    plt.title("Make% by Distance and Shooter Tier")
    plt.legend(title="SHOOTER_TIER")
    plt.tight_layout()
    plt.savefig("tier_distance_make_pct.png", dpi=200)
    plt.close()


def time_phase_plot(df):
    t = df["TIME_REMAINING"]

    conds = []
    labels = []

    conds.append(t >= 6 * 60)
    labels.append("Early (12-6 min)")

    conds.append((t < 6 * 60) & (t >= 2 * 60))
    labels.append("Middle (6-2 min)")

    conds.append(t < 2 * 60)
    labels.append("Late (last 2 min)")

    # IMPORTANT: default is a string so dtype is consistent
    df["TIME_PHASE"] = np.select(conds, labels, default="other")

    tmp = df.groupby("TIME_PHASE")["SHOT_MADE_FLAG"].agg(["mean", "count"]).reset_index()
    print("\nMake% by time in period:")
    print(tmp)

    plt.figure(figsize=(6, 5))
    plt.bar(tmp["TIME_PHASE"], tmp["mean"])
    plt.ylim(0, 1)
    plt.xlabel("Time in Period")
    plt.ylabel("Make Percentage")
    plt.title("Make% by Time in Period")
    plt.tight_layout()
    plt.savefig("time_phase_make_pct.png", dpi=200)
    plt.close()


def simple_shot_advisor(model):
    fake_row = {
        "LOC_X": 0,
        "LOC_Y": 235,
        "SHOT_DISTANCE": 24,
        "PERIOD": 2,
        "MINUTES_REMAINING": 6,
        "SECONDS_REMAINING": 0,
        "SHOT_TYPE": "3PT Field Goal",
        "SHOT_ZONE_BASIC": "Above the Break 3",
        "PLAYER_FG_PCT": 0.40,
        "PLAYER_SHOT_ATTEMPTS": 300,
        "ANGLE": np.arctan2(235, 0 + 1e-9),
        "TIME_REMAINING": 6 * 60,
        "DIST_CAT": 5,
        "IS_CORNER": 0,
        "SHOOTER_TIER": "top",
    }

    df_top = pd.DataFrame([fake_row])

    fake_row_bottom = dict(fake_row)
    fake_row_bottom["PLAYER_FG_PCT"] = 0.30
    fake_row_bottom["PLAYER_SHOT_ATTEMPTS"] = 100
    fake_row_bottom["SHOOTER_TIER"] = "bottom"

    df_bottom = pd.DataFrame([fake_row_bottom])

    p_top = model.predict_proba(df_top)[0, 1]
    p_bottom = model.predict_proba(df_bottom)[0, 1]

    print("\n=== Simple Shot Advisor Demo ===")
    print("Scenario: Above-the-break 3, top of the key, 24 ft, Q2, 6:00 remaining")
    print("Top-tier shooter make probability:   {:.2f}".format(p_top))
    print("Bottom-tier shooter make probability:{:.2f}".format(p_bottom))


if __name__ == "__main__":
    data = load_data()

    make_shot_chart(data)

    X, y, pre, num_cols, cat_cols = get_xy(data)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    models = build_models(pre)

    acc_df = model_accuracy_plots(models, X_tr, X_te, y_tr, y_te)

    rf_pipe = models["Random Forest"]

    rf_feature_importance_plot(rf_pipe, X_te, y_te, None)

    build_player_xfg(data, rf_pipe, X, y)

    tier_distance_plots(data)

    time_phase_plot(data)

    simple_shot_advisor(rf_pipe)
