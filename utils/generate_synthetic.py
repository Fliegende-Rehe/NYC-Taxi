"""
Generates a synthetic NYC-like taxi CSV for one month and writes CSV to ./data/
Usage: python generate_synthetic.py --rows 100000 --out data/yellow_tripdata_2022-01_sample.csv
"""
import argparse
import os
import math
import random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

random.seed(42)
np.random.seed(42)

parser = argparse.ArgumentParser()
parser.add_argument('--rows', type=int, default=100000)
parser.add_argument('--out', type=str, default='data/yellow_tripdata_2022-01_sample.csv')
args = parser.parse_args()

os.makedirs(os.path.dirname(args.out), exist_ok=True)

n = args.rows
start = datetime(2022,1,1)
end = datetime(2022,1,31,23,59,59)
total_seconds = int((end - start).total_seconds())

def random_timestamp():
    return start + timedelta(seconds=random.randint(0, total_seconds))

min_lat, max_lat = 40.55, 40.92
min_lon, max_lon = -74.15, -73.70

def haversine(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2-lat1)
    dlambda = math.radians(lon2-lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    return 2*R*math.asin(math.sqrt(a))

rows = []
for i in range(n):
    pickup_dt = random_timestamp()
    duration_min = max(1, int(np.random.exponential(scale=12)))
    dropoff_dt = pickup_dt + timedelta(minutes=duration_min)
    pu_lat = random.uniform(min_lat, max_lat)
    pu_lon = random.uniform(min_lon, max_lon)
    do_lat = pu_lat + np.random.normal(scale=0.01)
    do_lon = pu_lon + np.random.normal(scale=0.01)
    distance_km = max(0.05, haversine(pu_lat, pu_lon, do_lat, do_lon))
    base_fare = 2.5 + distance_km * 1.8 + duration_min * 0.2 + np.random.normal(scale=1.0)
    fare_amount = round(max(3.0, base_fare), 2)
    tip = round(max(0.0, np.random.exponential(scale=1.5)), 2)
    total_amount = fare_amount + tip
    passenger_count = np.random.choice([1,1,1,2,3,4], p=[0.6,0.1,0.1,0.1,0.05,0.05])
    vendor_id = np.random.choice([1,2])
    payment_type = np.random.choice(['card','cash'], p=[0.85,0.15])
    rows.append({
        "pickup_datetime": pickup_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "dropoff_datetime": dropoff_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "pickup_latitude": pu_lat,
        "pickup_longitude": pu_lon,
        "dropoff_latitude": do_lat,
        "dropoff_longitude": do_lon,
        "trip_distance_km": round(distance_km,3),
        "fare_amount": fare_amount,
        "tip_amount": tip,
        "total_amount": round(total_amount,2),
        "passenger_count": passenger_count,
        "vendor_id": vendor_id,
        "payment_type": payment_type
    })

pd.DataFrame(rows).to_csv(args.out, index=False)
print('Wrote', args.out)
