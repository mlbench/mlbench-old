package MLbenchmark.solvers

import org.apache.spark.SparkContext
import org.apache.spark.mllib.regression.{RidgeRegressionWithSGD, LabeledPoint, RidgeRegressionModel}
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.optimization.{LBFGS, LogisticGradient, SquaredL2Updater}
import org.apache.spark.rdd.RDD
import MLbenchmark.utils._
import breeze.linalg.DenseVector


object SGD{
	def run_SGD(trainData: RDD[MLbenchmark.utils.LabeledPoint], testData: RDD[MLbenchmark.utils.LabeledPoint], Iter:Int)
	{
		val numFeatures = trainData.take(1)(0).features.size
		val initialWeights = Vectors.dense(new Array[Double](numFeatures))
		//val ridge_regression = new RidgeRegressionWithSGD()
		val training = trainData.map(point => org.apache.spark.mllib.regression.LabeledPoint(point.label, Vectors.sparse(numFeatures, point.features.index, point.features.data))).cache()
		val model = RidgeRegressionWithSGD.train(training, Iter,1.0, 1.0, 0.1)
		val weightsVector = new DenseVector(model.weights.toArray)



		val valAndPreds = testData.map{ 
			point => 
			val prediction = model.predict(Vectors.sparse(numFeatures, point.features.index, point.features.data))
      	(point.label, prediction)
		}
		val MSE = valAndPreds.map{case(v,p) => math.pow((v-p),2)}.mean()
		println("SGD training Mean Squared Error = " + MSE)
	}
}