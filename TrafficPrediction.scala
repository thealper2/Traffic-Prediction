import org.apache.spark.sql._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.mllib.evaluation.RegressionMetrics
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.regression.LinearRegressionModel
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.tuning.CrossValidator
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.ml.evaluation.RegressionEvaluator

object TrafficPrediction {
	def main(): Unit = {
		//val spark = SparkSession.builder
		//	.master("local[4]")
		//	.appName("TrafficPrediction")
		//	.getOrCreate()

		import spark.implicits._

		val df_path = "/home/alper/Spark/Scala/UrbanTrafficPrediction/Urban.csv"

		val manual_schema = StructType(Array(
			StructField("Hour (Coded)", IntegerType, true),
			StructField("Immobilized bus", IntegerType, true),
			StructField("Broken Truck", IntegerType, true),
			StructField("Vehicle excess", IntegerType, true),
			StructField("Accident victim", IntegerType, true),
			StructField("Running over", IntegerType, true),
			StructField("Fire vehicles", IntegerType, true),
			StructField("Occurrence involving freight", IntegerType, true),
			StructField("Incident involving dangerous freight", IntegerType, true),
			StructField("Lack of electricity", IntegerType, true),
			StructField("Fire", IntegerType, true),
			StructField("Point of flooding", IntegerType, true),
			StructField("Manifestations", IntegerType, true),
			StructField("Defect in the network of trolleybuses", IntegerType, true),
			StructField("Tree on the road", IntegerType, true),
			StructField("Semaphore off", IntegerType, true),
			StructField("Intermittent Semaphore", IntegerType, true),
			StructField("Slowness in traffic (%)", StringType, true),
		))

		var df = spark.read
			.option("header", true)
			.option("inferSchema", true)
			.option("delimiter", ";")
			.schema(manual_schema)
			.csv(df_path)
			.withColumn("Slowness in traffic (%)", regexp_replace(col("Slowness in traffic (%)"), ",", "."))

		df = df.withColumn("Slowness in traffic (%)", col("Slowness in traffic (%)").cast("double"))

		df.show(10)
		df.printSchema()
		df.describe().show()

		df = df.withColumnRenamed("Slowness in traffic (%)", "label")

		val vector_assembler = new VectorAssembler()
			.setInputCols(df.columns.dropRight(1))
			.setOutputCol("features")

		val assembleDF = vector_assembler.transform(df)
			.select("features", "label")

		assembleDF.show(10)

		val seed = 4242
		val splits = assembleDF.randomSplit(Array(0.8, 0.2), seed)
		val (train_df, test_df) = (splits(0), splits(1))

		train_df.cache
		test_df.cache

		println("Train: ", train_df.count())
		println("Test: ", test_df.count())

		val lr_object = new LinearRegression()
			.setFeaturesCol("features")
			.setLabelCol("label")

		val lr_model = lr_object.fit(train_df)

		val trainPredLabel = lr_model.transform(test_df)
			.select("label", "prediction")
			.map{case Row(label: Double, prediction: Double) => (label, prediction)}.rdd

		var metrics1 = new RegressionMetrics(trainPredLabel)

		println("----- LR -----")
		println("MAE: " + metrics1.meanAbsoluteError)
		println("MSE: " + metrics1.meanSquaredError)
		println("RMSE: " + metrics1.rootMeanSquaredError)
		println("R^2: " + metrics1.r2)

		val maxIter = Array(10, 25, 50, 100, 250, 500)
		val regParam = Array(0.001, 0.01, 0.1)
		val tol = Array(0.01, 0.1)
		val numFolds = 10
		val paramGrid = new ParamGridBuilder()
			.addGrid(lr_object.maxIter, maxIter)
			.addGrid(lr_object.regParam, regParam)
			.addGrid(lr_object.tol, tol)
			.build()

		val evaluator = new RegressionEvaluator()
		val cv = new CrossValidator()
			.setEstimator(lr_object)
			.setEvaluator(evaluator)
			.setEstimatorParamMaps(paramGrid)
			.setNumFolds(numFolds)

		val cv_model = cv.fit(train_df)
		val cvPredLabel = cv_model.transform(test_df)
			.select("label", "prediction")
			.map{case Row(label: Double, prediction: Double) => (label, prediction)}.rdd

		var metrics2 = new RegressionMetrics(cvPredLabel)

		println("----- CV -----")
		println("MAE: " + metrics2.meanAbsoluteError)
		println("MSE: " + metrics2.meanSquaredError)
		println("RMSE: " + metrics2.rootMeanSquaredError)
		println("R^2: " + metrics2.r2)

		//spark.stop()
	}
}
