# nba_loc_visualization.py
#
# Creates multiple plots to help explain the model and data:
# 1) Basic shot chart
# 2) Shot distance histogram
# 3) Make percentage by court zone
# 4) Make percentage by distance range
# 5) Corner 3 vs non-corner 3 make rates

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("nba_bin_df.csv")

# -------------------------------------------------------------------
# 1. Shot chart (same idea as before)
# -------------------------------------------------------------------
plt.figure(figsize=(6, 6))
sns.scatterplot(
    data=df.sample(min(5000, len(df))), 
    x="LOC_X",
    y="LOC_Y",
    hue="SHOT_MADE_FLAG",
    palette={0: "red", 1: "green"},
    s=10,
    alpha=0.4,
)
plt.title("Shot Chart (2025 Season)")
plt.xlabel("LOC_X")
plt.ylabel("LOC_Y")
plt.legend(title="Made?", labels=["Miss (0)", "Make (1)"])
plt.tight_layout()
plt.savefig("shot_chart.png", dpi=200)
plt.close()

# -------------------------------------------------------------------
# 2. Shot distance histogram
# -------------------------------------------------------------------
plt.figure(figsize=(6, 4))
plt.hist(df["SHOT_DISTANCE"], bins=30, edgecolor="black")
plt.title("Shot Distance Distribution")
plt.xlabel("Distance (feet)")
plt.ylabel("Number of shots")
plt.tight_layout()
plt.savefig("distance_histogram.png", dpi=200)
plt.close()

# -------------------------------------------------------------------
# 3. Make percentage by basic court zone
# -------------------------------------------------------------------
zone_pct = (
    df.groupby("SHOT_ZONE_BASIC")["SHOT_MADE_FLAG"]
    .mean()
    .sort_values()
)

plt.figure(figsize=(8, 4))
sns.barplot(x=zone_pct.index, y=zone_pct.values)
plt.xticks(rotation=45, ha="right")
plt.ylabel("Make percentage")
plt.ylim(0, 1)
plt.title("Shot Make Percentage by SHOT_ZONE_BASIC")
plt.tight_layout()
plt.savefig("zone_make_percentage.png", dpi=200)
plt.close()

# -------------------------------------------------------------------
# 4. Make percentage by distance range
# -------------------------------------------------------------------
dist_bins = [0, 5, 10, 15, 20, 25, 30, 35, 40]
dist_labels = [f"{dist_bins[i]}-{dist_bins[i+1]}" for i in range(len(dist_bins) - 1)]

df["DIST_BIN"] = pd.cut(
    df["SHOT_DISTANCE"],
    bins=dist_bins,
    labels=dist_labels,
    include_lowest=True,
)

dist_pct = (
    df.groupby("DIST_BIN")["SHOT_MADE_FLAG"]
    .mean()
    .reindex(dist_labels)
)

plt.figure(figsize=(8, 4))
sns.lineplot(x=dist_pct.index, y=dist_pct.values, marker="o")
plt.xticks(rotation=45, ha="right")
plt.ylabel("Make percentage")
plt.ylim(0, 1)
plt.title("Shot Make Percentage by Distance Range")
plt.tight_layout()
plt.savefig("distance_make_percentage.png", dpi=200)
plt.close()

# -------------------------------------------------------------------
# 5. Corner 3 vs non-corner 3 make rates
# -------------------------------------------------------------------
corner_mask = (
    (df["SHOT_TYPE"] == "3PT Field Goal")
    & (df["LOC_Y"] < 92)
    & (df["LOC_X"].abs() > 220)
)

df["IS_CORNER_THREE"] = corner_mask.astype(int)

corner_pct = (
    df[df["SHOT_TYPE"] == "3PT Field Goal"]
    .groupby("IS_CORNER_THREE")["SHOT_MADE_FLAG"]
    .mean()
)

labels = ["Non-corner 3", "Corner 3"]
values = [corner_pct.get(0, 0.0), corner_pct.get(1, 0.0)]

plt.figure(figsize=(5, 4))
sns.barplot(x=labels, y=values)
plt.ylim(0, 1)
plt.ylabel("Make percentage")
plt.title("Corner 3 vs Non-corner 3 Make Percentage")
plt.tight_layout()
plt.savefig("corner_vs_noncorner_3s.png", dpi=200)
plt.close()
