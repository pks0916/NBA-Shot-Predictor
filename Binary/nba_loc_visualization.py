# nba_loc_visualization.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("nba_bin_df.csv")

# Shot chart
plt.figure(figsize=(6,6))
sns.scatterplot(
    data=df.sample(min(5000, len(df))), 
    x="LOC_X", y="LOC_Y",
    hue="SHOT_MADE_FLAG",
    palette={0:"red", 1:"green"},
    s=10, alpha=0.4
)
plt.title("Shot Chart (2025 Season)")
plt.savefig("shot_chart.png", dpi=200)
plt.close()

# Distance histogram
plt.figure(figsize=(6,4))
plt.hist(df["SHOT_DISTANCE"], bins=30, edgecolor="black")
plt.title("Shot Distance Distribution")
plt.xlabel("Distance (feet)")
plt.ylabel("Frequency")
plt.savefig("distance_histogram.png", dpi=200)
plt.close()
