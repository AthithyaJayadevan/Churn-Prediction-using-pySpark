import pandas
import matplotlib as mp
from pyspark.sql import SQLContext, SparkSession
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import UserDefinedFunction
from pyspark.ml.linalg import Vectors
from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, VectorIndexer
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

spark = SparkSession.builder \
         .appName("Project") \
         .master("local")   \
         .config("spark.executor.memory", "2gb") \
         .getOrCreate()
sc = spark.sparkContext
sqlc = SQLContext(sc)

train = sqlc.read.load("C:\Users\jayad\Downloads\churn-bigml-80.csv", format="com.databricks.spark.csv", header="true", inferschema="true")
test = sqlc.read.load("C:\Users\jayad\Downloads\churn-bigml-20.csv", format="com.databricks.spark.csv", header="true", inferschema="true")

bin_map = {'Yes': 1.0, 'No': 0.0, True: 1.0, False: 0.0}
To_Num = UserDefinedFunction(lambda k: bin_map[k], DoubleType())

train = train.drop('State').drop('Area code') \
    .drop('Total day charge').drop('Total eve charge') \
    .drop('Total night charge').drop('Total intl charge') \
    .withColumn('International plan', To_Num(train['International plan'])) \
    .withColumn('Churn', To_Num(train['Churn'])) \
    .withColumn('Voice mail plan', To_Num(train['Voice mail plan']))

test = test.drop('State').drop('Area code') \
    .drop('Total day charge').drop('Total eve charge') \
    .drop('Total night charge').drop('Total intl charge') \
    .withColumn('International plan', To_Num(test['International plan'])) \
    .withColumn('Churn', To_Num(test['Churn'])) \
    .withColumn('Voice mail plan', To_Num(test['Voice mail plan']))


def vectorizedataframe(data):
    return data.rdd.map(lambda r:  [r[-1], Vectors.dense(r[:-1])]).toDF(['label', 'features'])


Vectored_data = vectorizedataframe(train)
lab_indexer = StringIndexer(inputCol='label', outputCol='indexed_label').fit(Vectored_data)
feat_indexer = VectorIndexer(inputCol='features', outputCol='indexed_features', maxCategories=3).fit(Vectored_data)
dec_tree = DecisionTreeClassifier(labelCol='indexed_label', featuresCol='indexed_features')
pipeline = Pipeline(stages=[lab_indexer, feat_indexer, dec_tree])

param_grid_model = ParamGridBuilder().addGrid(dec_tree.maxDepth, [2, 3, 4, 5, 6, 7, 8]).build()

evaluator_met = MulticlassClassificationEvaluator(labelCol='indexed_label', predictionCol='prediction', metricName='f1')
cross_validator = CrossValidator(estimator=pipeline, estimatorParamMaps=param_grid_model, evaluator=evaluator_met , numFolds=3)

cv_model = cross_validator.fit(Vectored_data)

Vectored_test_data = vectorizedataframe(test)
trans_data = cv_model.transform(Vectored_test_data)

trans_data.show(5)
print(evaluator_met.getMetricName(), 'accuracy:', evaluator_met.evaluate(trans_data))






