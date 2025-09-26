import argparse, os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib, math
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--in", dest="infile", required=True)
parser.add_argument("--out", dest="outdir", default="models")
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)

agg = pd.read_csv(args.infile, parse_dates=["hour_ts"])
agg = agg.sort_values(["zone_id", "hour_ts"])
agg["prev_1h"] = agg.groupby("zone_id")["pickups"].shift(1).fillna(0)
agg["prev_3h_mean"] = (
    agg.groupby("zone_id")["pickups"]
    .shift(1)
    .rolling(window=3, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
)
agg["hour"] = agg["hour_ts"].dt.hour
agg["weekday"] = agg["hour_ts"].dt.dayofweek
agg["target"] = agg.groupby("zone_id")["pickups"].shift(-1)
agg = agg.dropna(subset=["target"])

feature_cols = ["pickups", "prev_1h", "prev_3h_mean", "hour", "weekday"]
X = agg[feature_cols]
y = agg["target"]
split_idx = int(len(X) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]


try:
    import xgboost as xgb

    model = xgb.XGBRegressor(n_estimators=200, max_depth=6, random_state=42)
    model.fit(X_train, y_train)
    model_used = "XGBoost"
except Exception:
    model = RandomForestRegressor(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    model_used = "RandomForest"

preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)
rmse = math.sqrt(mean_squared_error(y_test, preds))
r2 = r2_score(y_test, preds)
print("Model:", model_used, "MAE:", mae, "RMSE:", rmse, "R2:", r2)


joblib.dump(model, os.path.join(args.outdir, "pickup_model.joblib"))
print("Saved model to", os.path.join(args.outdir, "pickup_model.joblib"))


plt.figure(figsize=(6, 6))
plt.scatter(y_test, preds, alpha=0.3)
plt.xlabel("Actual pickups next hour")
plt.ylabel("Predicted pickups")
plt.title(f"{model_used} Predictions vs Actuals")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.savefig(os.path.join(args.outdir, "pred_vs_actual.png"))
plt.close()

if hasattr(model, "feature_importances_"):
    importances = model.feature_importances_
    feat_df = pd.DataFrame(
        {"feature": feature_cols, "importance": importances}
    ).sort_values("importance", ascending=False)
    feat_df.plot(x="feature", y="importance", kind="bar", title="Feature Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(args.outdir, "feature_importance.png"))
    plt.close()
    feat_df.to_csv(os.path.join(args.outdir, "feature_importance.csv"), index=False)
