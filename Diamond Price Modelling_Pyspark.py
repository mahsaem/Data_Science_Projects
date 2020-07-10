# Databricks notebook source
# MAGIC %md In this notebook, I'm going to build a model to predict the price of a diamond based on the available features, using the Apache Spark ML pipeline.
# MAGIC 
# MAGIC Information about the dataset:
# MAGIC 
# MAGIC http://ggplot2.tidyverse.org/reference/diamonds.html

# COMMAND ----------

dataPath = "/databricks-datasets/Rdatasets/data-001/csv/ggplot2/diamonds.csv"
diamonds = sqlContext.read.format("com.databricks.spark.csv").option("header","true").option("inferSchema", "true").load(dataPath)

# COMMAND ----------

diamonds.printSchema()

# COMMAND ----------

diamonds.show()

# COMMAND ----------

# MAGIC %fs ls /databricks-datasets/Rdatasets/data-001/csv/ggplot2/

# COMMAND ----------

#drop the id column, and get rid of nas
df_no_id = diamonds.drop('_c0')
df_no_na = df_no_id.dropna()

# COMMAND ----------

df = df_no_na.select('price', 'carat', 'cut', 'color', 'clarity', 
  'depth', 'table', 'x', 'y', 'z')
df.show()

# COMMAND ----------

df = df.withColumnRenamed('price', 'label')

# COMMAND ----------

df.show()

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler

cutIndexer = StringIndexer(inputCol='cut', outputCol='cutIndex')
colorIndexer = StringIndexer(inputCol='color', outputCol='colorIndex')
clarityIndexer = StringIndexer(inputCol='clarity', outputCol='clarityIndex')

df = cutIndexer.fit(df).transform(df)
df = colorIndexer.fit(df).transform(df)
df = clarityIndexer.fit(df).transform(df)

df.show()

# COMMAND ----------

from pyspark.ml.feature import OneHotEncoderEstimator

OHE = OneHotEncoderEstimator(inputCols=['cutIndex', 'colorIndex', 'clarityIndex'],outputCols=['cut_OHE', 'color_OHE', 'clarity_OHE'])

df = OHE.fit(df).transform(df)

# COMMAND ----------

df.show()

# COMMAND ----------

assembler = VectorAssembler(
  inputCols= ['carat', 'depth', 'table', 'x', 'y', 'z', 'cut_OHE', 'color_OHE', 'clarity_OHE'], outputCol=('features_assem'))

df = df.dropna()

# COMMAND ----------

from pyspark.ml.feature import MinMaxScaler

scaler = MinMaxScaler(inputCol="features_assem", outputCol="scaledFeatures")
pipeline = Pipeline(stages=[assembler, scaler])
scalerModel = pipeline.fit(df)
scaled_df = scalerModel.transform(df)
display(scaled_df)

# COMMAND ----------

training, test = scaled_df.randomSplit([0.7, 0.3])
training.cache()
test.cache()

# COMMAND ----------

from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.evaluation import RegressionEvaluator

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator

dt = DecisionTreeRegressor(featuresCol = "scaledFeatures")

pipeline = Pipeline(stages= [dt])

paramGrid = ParamGridBuilder() \
    .addGrid(dt.maxDepth, [5, 10, 15, 20, 30]) \
    .addGrid(dt.maxBins, [10, 20, 30, 50]) \
    .build()

# COMMAND ----------

cv = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=RegressionEvaluator(),
                          numFolds=3)
cvModel = cv.fit(training)
predictions = cvModel.transform(test)

# COMMAND ----------

evaluator = RegressionEvaluator(labelCol = "label", predictionCol= "prediction", metricName = "rmse")

rmse = evaluator.evaluate(predictions)

evaluator_r2 = RegressionEvaluator(labelCol = "label", predictionCol= "prediction", metricName = "r2")

r2 = evaluator_r2.evaluate(predictions)

evaluator_mae = RegressionEvaluator(labelCol = "label", predictionCol= "prediction", metricName = "mae")

mae = evaluator_mae.evaluate(predictions)

evaluator_mse = RegressionEvaluator(labelCol = "label", predictionCol= "prediction", metricName = "mse")

mse = evaluator_mse.evaluate(predictions)


print("RMSE on test data = ", rmse)
print("R_squared on test data = ", r2)
print("Mean Absolute Error (MAE) on test data = ", mae)
print("Mean Squared Error (MSE) on test data = ", mse)

predictions.select("label", "prediction").show()

# COMMAND ----------

from pyspark.ml.regression import RandomForestRegressor

rf = (RandomForestRegressor()
          .setLabelCol('label')
          .setFeaturesCol('scaledFeatures'))
#stages = [indexers , encoders, assembler_1 , assembler, scaler, rf]
pipeline = Pipeline(stages=[rf])

paramGrid = (ParamGridBuilder()
            .addGrid(rf.maxDepth, [5, 10])
            .addGrid(rf.numTrees, [10, 20])
            .addGrid(rf.maxBins, [10, 20, 30, 50])
            .build())

# COMMAND ----------

cv = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=RegressionEvaluator(),
                          numFolds=3)
cvModel = cv.fit(training)
predictions = cvModel.transform(test)

# COMMAND ----------

evaluator = RegressionEvaluator(labelCol = "label", predictionCol= "prediction", metricName = "rmse")

rmse = evaluator.evaluate(predictions)

evaluator_r2 = RegressionEvaluator(labelCol = "label", predictionCol= "prediction", metricName = "r2")

r2 = evaluator_r2.evaluate(predictions)

evaluator_mae = RegressionEvaluator(labelCol = "label", predictionCol= "prediction", metricName = "mae")

mae = evaluator_mae.evaluate(predictions)

evaluator_mse = RegressionEvaluator(labelCol = "label", predictionCol= "prediction", metricName = "mse")

mse = evaluator_mse.evaluate(predictions)


print("RMSE on test data = ", rmse)
print("R_squared on test data = ", r2)
print("Mean Absolute Error (MAE) on test data = ", mae)
print("Mean Squared Error (MSE) on test data = ", mse)

predictions.select("label", "prediction").show()

# COMMAND ----------


