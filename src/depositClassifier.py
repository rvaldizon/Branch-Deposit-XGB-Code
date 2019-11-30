from pyspark.sql import SparkSession
from pyspark import SparkConf, SparkContext
from pyspark.ml import Pipeline
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.feature import OneHotEncoderEstimator, VectorAssembler, StringIndexer
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

#conf =SparkConf().setAppName("bankClassifier").setMaster("local[*]")
#spark = SparkContext(conf = conf)

spark = SparkSession.builder.getOrCreate()
data = spark.read.csv("../data/bank.csv", header = True, inferSchema = True)

cat = [i[0] for i in data.dtypes if i[1]=="string" and i[0]!= "deposit"]
num = [i[0] for i in data.dtypes if i[1]!="string" and i[0]!= "deposit"]

stages = []

for i in cat:
    strToNum = StringIndexer(inputCol = i, outputCol = i+"_index")
    vector = OneHotEncoderEstimator(inputCols = [strToNum.getOutputCol()], outputCols =[i + "_vector"])
    stages += [strToNum, vector]

assemblerInputs = [i +"_vector" for i in cat] + num
vectorAssembler = VectorAssembler(inputCols = assemblerInputs, outputCol = "features")

label = StringIndexer(inputCol = "deposit", outputCol = "label")

stages += [vectorAssembler, label]

data = Pipeline(stages = stages).fit(data).transform(data).select("features", "label")

train, test = data.randomSplit([0.7,0.3])

gbt = GBTClassifier()

params = ParamGridBuilder().addGrid(gbt.maxDepth, [2,4,6]).build()

evaluator = BinaryClassificationEvaluator()

model = CrossValidator(estimator = gbt, estimatorParamMaps = params, evaluator = evaluator, numFolds = 8).fit(train)

model.save("gbt_model")
