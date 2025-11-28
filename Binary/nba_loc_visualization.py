import numpy as np
import pandas as pd

import matplotlib.pyplot as plt



df = pd.read_csv("shot_charts_2025_enhanced.csv")



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



plt.figure(figsize=(6,4))
plt.hist(df["SHOT_DISTANCE"], bins=30)
plt.xlabel("SHOT_DISTANCE (ft)")
plt.ylabel("count")
plt.title("Shot Distance Histogram")
plt.tight_layout()
plt.savefig("shot_distance_hist.png", dpi=200)
plt.close()
