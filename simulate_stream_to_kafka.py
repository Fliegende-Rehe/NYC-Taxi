import argparse
import time
import pandas as pd
from kafka import KafkaProducer
import json

parser = argparse.ArgumentParser()
parser.add_argument('--csv', required=True)
parser.add_argument('--topic', required=True)
parser.add_argument('--brokers', required=True)
parser.add_argument('--delay', type=float, default=0.1)
args = parser.parse_args()

df = pd.read_csv(args.csv, parse_dates=['pickup_datetime'])
producer = KafkaProducer(bootstrap_servers=args.brokers.split(','),
                         value_serializer=lambda v: json.dumps(v).encode('utf-8'))

for _, row in df.iterrows():
    record = {
        "pickup_datetime": row['pickup_datetime'].isoformat(),
        "pickup_latitude": row['pickup_latitude'],
        "pickup_longitude": row['pickup_longitude']
    }
    producer.send(args.topic, value=record)
    time.sleep(args.delay)

producer.flush()
print("Streaming finished")
