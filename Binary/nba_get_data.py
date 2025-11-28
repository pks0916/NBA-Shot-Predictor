import pandas as pd


input_file  = "../shot_charts_2025.csv"          # raw file (one level up)
output_file = "shot_charts_2025_enhanced.csv"    # cleaned file (inside Binary)


df = pd.read_csv(input_file)


df = df[df["SHOT_ATTEMPTED_FLAG"] == 1].copy()

df["SHOT_MADE_FLAG"] = df["SHOT_MADE_FLAG"].astype(int)


if "PLAYER_ID" not in df.columns:
    raise KeyError("PLAYER_ID column missing in shot_charts_2025.csv")

if "PLAYER_NAME" not in df.columns:
    raise KeyError("PLAYER_NAME column missing in shot_charts_2025.csv")



player_stats = df.groupby("PLAYER_ID")["SHOT_MADE_FLAG"].agg(["mean", "count"])
player_stats = player_stats.reset_index()
player_stats = player_stats.rename(
    columns={"mean": "PLAYER_FG_PCT", "count": "PLAYER_SHOT_ATTEMPTS"}
)


q1 = player_stats["PLAYER_FG_PCT"].quantile(1.0 / 3.0)
q2 = player_stats["PLAYER_FG_PCT"].quantile(2.0 / 3.0)


def tier_func(x):
    if x <= q1:
        return "bottom"
    elif x <= q2:
        return "middle"
    else:
        return "top"



player_stats["SHOOTER_TIER"] = player_stats["PLAYER_FG_PCT"].apply(tier_func)


df = df.merge(
    player_stats[["PLAYER_ID", "PLAYER_FG_PCT", "PLAYER_SHOT_ATTEMPTS", "SHOOTER_TIER"]],
    on="PLAYER_ID",
    how="left",
)



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
    "PLAYER_NAME",
    "PLAYER_FG_PCT",
    "PLAYER_SHOT_ATTEMPTS",
    "SHOOTER_TIER",
    "SHOT_MADE_FLAG",
]

df = df[keep_cols].dropna()



df.to_csv(output_file, index=False)

print("saved", output_file)
print("shape:", df.shape)
print("tier cutoffs:", round(q1, 3), round(q2, 3))
print(df.head(5))
