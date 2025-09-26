import pandas as pd
import joblib

df = pd.read_csv('output/agg_zone_hour.csv', parse_dates=['hour_ts'])

df = df.sort_values(['zone_id','hour_ts'])
df['prev_1h'] = df.groupby('zone_id')['pickups'].shift(1).fillna(0)
df['prev_3h_mean'] = df.groupby('zone_id')['pickups'].shift(1).rolling(window=3, min_periods=1).mean().reset_index(level=0, drop=True)
df['hour'] = df['hour_ts'].dt.hour
df['weekday'] = df['hour_ts'].dt.dayofweek

model = joblib.load('models/pickup_model.joblib')

feature_cols = ['pickups','prev_1h','prev_3h_mean','hour','weekday']
df['predicted_next_hour_pickups'] = model.predict(df[feature_cols])
df.to_csv('output/predictions.csv', index=False)
