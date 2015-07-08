package MLbenchmark.solvers

import org.apache.spark.SparkContext
import org.apache.spark.mllib.regression.{RidgeRegressionWithSGD, LabeledPoint, RidgeRegressionModel}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.optimization.{LBFGS, LogisticGradient, SquaredL2Updater}
import org.apache.spark.rdd.RDD

object SGD{
	def run_SGD(train_data: RDD[LabeledPoint], test_data: RDD[LabeledPoint])
	{
		val ridge_regression = new RidgeRegressionWithSGD(1.0, 100, 0.1, 1.0)
		val model = ridge_regression.run(train_data)
		val valAndPreds = test_data.map{ 
			point => 
			val prediction = model.predict(point.features)
			(point.label, prediction)
		}
		val MSE = valAndPreds.map{case(v,p) => math.pow((v-p),2)}.mean()
		println("SGD training Mean Squared Error = " + MSE)
	}
}