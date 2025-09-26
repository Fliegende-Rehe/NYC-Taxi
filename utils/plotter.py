import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

sns.set(style="whitegrid")

# --- STREAM PREDICTIONS ---
stream_dir = Path("./stream_predictions")
stream_col_names = ["zone_id","hour_ts","pickups","prev_1h","prev_3h_mean","hour","predicted_next_hour_pickups"]

stream_dfs = []
for f in stream_dir.glob("part-*.csv"):
    if f.stat().st_size == 0:  # skip empty files
        continue
    df = pd.read_csv(f, names=stream_col_names, parse_dates=["hour_ts"])
    stream_dfs.append(df)

if stream_dfs:
    df_stream = pd.concat(stream_dfs, ignore_index=True)
    # Example plot: predicted vs actual pickups over time
    plt.figure(figsize=(12,6))
    plt.plot(df_stream.groupby("hour_ts")["pickups"].sum(), label="Actual pickups")
    plt.plot(df_stream.groupby("hour_ts")["predicted_next_hour_pickups"].sum(), label="Predicted pickups")
    plt.title("Stream: Predicted vs Actual pickups")
    plt.xlabel("Time")
    plt.ylabel("Pickups")
    plt.legend()
    plt.savefig("stream_pred_vs_actual.png")
    plt.close()

# --- BATCH PREDICTIONS ---
df_batch = pd.read_csv("predictions.csv", parse_dates=["hour_ts"])

plt.figure(figsize=(12,6))
plt.plot(df_batch.groupby("hour_ts")["pickups"].sum(), label="Actual pickups")
plt.plot(df_batch.groupby("hour_ts")["predicted_next_hour_pickups"].sum(), label="Predicted pickups")
plt.title("Batch: Predicted vs Actual pickups")
plt.xlabel("Time")
plt.ylabel("Pickups")
plt.legend()
plt.savefig("pred_vs_actual.png")
plt.close()

# --- AGGREGATED DATA ---
df_agg = pd.read_csv("agg_zone_hour.csv", parse_dates=["hour_ts"])

plt.figure(figsize=(12,6))
df_agg.groupby("hour_ts")["pickups"].sum().plot()
plt.title("Aggregated pickups per hour")
plt.xlabel("Hour")
plt.ylabel("Pickups")
plt.savefig("agg_pickups_per_hour.png")
plt.close()

# --- FEATURE IMPORTANCE ---
df_feat = pd.read_csv("feature_importance.csv")

plt.figure(figsize=(8,5))
sns.barplot(x="importance", y="feature", data=df_feat.sort_values("importance", ascending=False))
plt.title("Feature Importance")
plt.savefig("feature_importance.png")
plt.close()

print("All plots generated successfully!")
