# nba_get_data.py
# Loads ../shot_charts_2025.csv and creates a cleaned dataframe with player-level stats.
# Adds:
# - PLAYER_FG_PCT (each player's season shooting percentage)
# - PLAYER_SHOT_ATTEMPTS
# - SHOOTER_TIER ("bottom", "middle", "top" based on FG%)
# Saves to: shot_charts_2025_enhanced.csv (inside Binary/)

import pandas as pd

# IMPORTANT: this script is in Binary/, the CSV is one folder up
INPUT_FILE = "../shot_charts_2025.csv"
OUTPUT_FILE = "shot_charts_2025_enhanced.csv"

# -----------------------------------------------------------
# 1. Load raw data
# -----------------------------------------------------------
df = pd.read_csv(INPUT_FILE)

# Keep only attempted shots
df = df[df["SHOT_ATTEMPTED_FLAG"] == 1].copy()

# Ensure target is int
df["SHOT_MADE_FLAG"] = df["SHOT_MADE_FLAG"].astype(int)

# Check that PLAYER_ID exists
if "PLAYER_ID" not in df.columns:
    raise KeyError(
        "Expected column 'PLAYER_ID' in shot_charts_2025.csv but did not find it."
    )

# -----------------------------------------------------------
# 2. Compute per-player FG% and shot attempts
# -----------------------------------------------------------
player_stats = (
    df.groupby("PLAYER_ID")["SHOT_MADE_FLAG"]
    .agg(["mean", "count"])
    .reset_index()
    .rename(columns={"mean": "PLAYER_FG_PCT", "count": "PLAYER_SHOT_ATTEMPTS"})
)

# Quantiles for bottom/middle/top 33%
q1 = player_stats["PLAYER_FG_PCT"].quantile(1 / 3)
q2 = player_stats["PLAYER_FG_PCT"].quantile(2 / 3)


def shooter_tier(pct):
    """Categorize shooters into bottom/middle/top tiers."""
    if pct <= q1:
        return "bottom"
    elif pct <= q2:
        return "middle"
    else:
        return "top"


player_stats["SHOOTER_TIER"] = player_stats["PLAYER_FG_PCT"].apply(shooter_tier)

# -----------------------------------------------------------
# 3. Merge shooter stats into main dataframe
# -----------------------------------------------------------
df = df.merge(
    player_stats[["PLAYER_ID", "PLAYER_FG_PCT", "PLAYER_SHOT_ATTEMPTS", "SHOOTER_TIER"]],
    on="PLAYER_ID",
    how="left",
)

# -----------------------------------------------------------
# 4. Keep only relevant columns
# -----------------------------------------------------------
keep_cols = [
    "LOC_X",
    "LOC_Y",
    "SHOT_DISTANCE",
    "PERIOD",
    "MINUTES_REMAINING",
    "SECONDS_REMAINING",
    "SHOT_TYPE",
    "SHOT_ZONE_BASIC",
    "PLAYER_ID",
    "PLAYER_FG_PCT",
    "PLAYER_SHOT_ATTEMPTS",
    "SHOOTER_TIER",
    "SHOT_MADE_FLAG",
]

df = df[keep_cols].dropna()

# -----------------------------------------------------------
# 5. Save processed data
# -----------------------------------------------------------
df.to_csv(OUTPUT_FILE, index=False)
print("✅ Saved:", OUTPUT_FILE)
print("Shape:", df.shape)
print("FG% quantile cutoffs:", round(q1, 3), round(q2, 3))
print(df.head(5))
