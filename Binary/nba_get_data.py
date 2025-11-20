# nba_get_data.py
# Loads shot_charts_2025.csv and creates a cleaned df used by all other scripts.

import pandas as pd

INPUT_FILE = "../shot_charts_2025.csv"
OUTPUT_FILE = "nba_bin_df.csv"

df = pd.read_csv(INPUT_FILE)

# Keep only attempted FGs
df = df[df["SHOT_ATTEMPTED_FLAG"] == 1].copy()

# Ensure target is int
df["SHOT_MADE_FLAG"] = df["SHOT_MADE_FLAG"].astype(int)

# Keep only columns your project uses
keep_cols = [
    "LOC_X", "LOC_Y", "SHOT_DISTANCE",
    "PERIOD", "MINUTES_REMAINING", "SECONDS_REMAINING",
    "SHOT_TYPE",
    "SHOT_ZONE_BASIC",
    "SHOT_MADE_FLAG"
]

df = df[keep_cols].dropna()

df.to_csv(OUTPUT_FILE, index=False)
print("Saved:", OUTPUT_FILE)
print(df.shape)
