import argparse, joblib
import pandas as pd
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import StructType, StructField, DoubleType, TimestampType
from pyspark.sql.functions import pandas_udf


parser = argparse.ArgumentParser()
parser.add_argument("--brokers", required=True)
parser.add_argument("--topic", required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()


model = joblib.load(args.model)


spark = SparkSession.builder.appName("TaxiStreamPredict").getOrCreate()


schema = StructType(
    [
        StructField("pickup_datetime", TimestampType()),
        StructField("pickup_latitude", DoubleType()),
        StructField("pickup_longitude", DoubleType()),
    ]
)


raw = (
    spark.readStream.format("kafka")
    .option("kafka.bootstrap.servers", args.brokers)
    .option("subscribe", args.topic)
    .load()
)

values = raw.selectExpr("CAST(value AS STRING)")
json_df = values.select(F.from_json(F.col("value"), schema).alias("data")).select(
    "data.*"
)


json_df = json_df.withWatermark("pickup_datetime", "1 hour")

agg = json_df.groupBy(F.window("pickup_datetime", "1 hour").alias("hour_window")).agg(
    F.count("*").alias("pickups")
)

agg = agg.withColumn("hour_ts", F.col("hour_window.start")).drop("hour_window")


agg = agg.withColumn("hour", F.hour("hour_ts"))
agg = agg.withColumn("weekday", F.dayofweek("hour_ts"))


agg = agg.withColumn("prev_1h", F.lit(0.0))
agg = agg.withColumn("prev_3h_mean", F.lit(0.0))


@pandas_udf("double")
def predict_udf(pdf: pd.DataFrame) -> pd.Series:

    features = pdf[["pickups", "prev_1h", "prev_3h_mean", "hour", "weekday"]].fillna(0)
    return pd.Series(model.predict(features))


agg = agg.withColumn(
    "predicted_next_hour_pickups",
    predict_udf(F.struct("pickups", "prev_1h", "prev_3h_mean", "hour", "weekday")),
)


query = (agg.writeStream
         .outputMode("append")
         .format("csv")
         .option("path", args.out)
         .option("checkpointLocation", args.out + "/_chk")
         .start())

query.awaitTermination(60)
