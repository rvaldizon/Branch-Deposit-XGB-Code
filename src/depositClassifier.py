from pyspark.sql import SparkSession
from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.classification import GBTClassifier
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoderEstimator, VectorAssembler, StringIndexer
from pyspark.ml.evaluation import BinaryClassificationEvaluator

conf =SparkConf().setAppName("bankClassifier").setMaster("local[*]")

spark = SparkContext(conf = conf)

#spark = SparkSession.builder.appName('abc').getOrCreate()
data = spark.read.csv("../data/bank.csv", header = True, inferSchema = True)
print(data.show())
