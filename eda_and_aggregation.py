import argparse, os
import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--in', dest='infile', required=True)
parser.add_argument('--out', dest='outdir', default='output')
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)

df = pd.read_csv(args.infile, parse_dates=['pickup_datetime'])
df['hour'] = df['pickup_datetime'].dt.hour
df['weekday'] = df['pickup_datetime'].dt.day_name()


hourly = df.groupby('hour').size().reset_index(name='pickups')
hourly.plot(x='hour', y='pickups', kind='line', title='Pickups by hour')
plt.savefig(os.path.join(args.outdir, 'pickups_by_hour.png'))
plt.close()


weekday = df.groupby('weekday').size().reindex(['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']).reset_index(name='pickups')
weekday.plot(x='weekday', y='pickups', kind='bar', title='Pickups by weekday')
plt.savefig(os.path.join(args.outdir, 'pickups_by_weekday.png'))
plt.close()


grid_x, grid_y = 20, 20

df['gx'] = pd.cut(df['pickup_longitude'], bins=grid_x, labels=False)
df['gy'] = pd.cut(df['pickup_latitude'], bins=grid_y, labels=False)
df['zone_id'] = df['gy'].astype(str) + '_' + df['gx'].astype(str)

agg = (df.groupby(['zone_id', df['pickup_datetime'].dt.floor('h').rename('hour_ts')])
       .size().reset_index(name='pickups'))

agg.to_csv(os.path.join(args.outdir, 'agg_zone_hour.csv'), index=False)
print('Saved aggegation to', os.path.join(args.outdir, 'agg_zone_hour.csv'))