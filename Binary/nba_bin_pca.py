import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler



df = pd.read_csv("shot_charts_2025_enhanced.csv")



df["ANGLE"] = np.arctan2(df["LOC_Y"], df["LOC_X"])

df["IS_CORNER"] = (
    (df["SHOT_TYPE"] == "3PT Field Goal")
    & (df["LOC_Y"] < 92)
    & (df["LOC_X"].abs() > 220)
).astype(int)

df["TIME_REMAINING"] = df["MINUTES_REMAINING"] * 60 + df["SECONDS_REMAINING"]

dist_bins = [0, 5, 10, 15, 20, 25, 30, 35, 40]
df["DIST_CAT"] = pd.cut(df["SHOT_DISTANCE"], bins=dist_bins, labels=False, include_lowest=True)
df["DIST_CAT"] = df["DIST_CAT"].fillna(len(dist_bins)).astype(int)

df["SHOOTER_TIER"] = df["SHOOTER_TIER"].astype(str)



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


Xt = pre.fit_transform(X)

if hasattr(Xt, "toarray"):
    Xt = Xt.toarray()



pca = PCA(n_components=2, random_state=42)

Z = pca.fit_transform(Xt)



print("explained variance ratio:", pca.explained_variance_ratio_)


pca_df = pd.DataFrame(
    {
        "pc1": Z[:, 0],
        "pc2": Z[:, 1],
        "SHOT_MADE_FLAG": y.values,
    }
)

pca_df.to_csv("nba_bin_df.csv", index=False)



plt.figure(figsize=(6,5))
colors = {0: "orange", 1: "steelblue"}

for val in [0, 1]:
    part = pca_df[pca_df["SHOT_MADE_FLAG"] == val]
    plt.scatter(part["pc1"], part["pc2"], s=4, alpha=0.4, c=colors[val], label=f"{val}")

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA of Shot Features (2D)")
plt.legend(title="SHOT_MADE_FLAG")
plt.tight_layout()
plt.savefig("nba_pca_scatter.png", dpi=200)
plt.close()
