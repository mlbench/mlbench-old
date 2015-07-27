package MLbenchmark.solvers

import org.apache.spark.SparkContext
import org.apache.spark.mllib.regression.{LassoWithSGD, RidgeRegressionWithSGD, LabeledPoint, RidgeRegressionModel}
import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.optimization.{LBFGS, LogisticGradient, SquaredL2Updater}
import org.apache.spark.rdd.RDD
import MLbenchmark.utils._
import breeze.linalg.DenseVector


object SGD{
	def run_SGD(trainData: RDD[MLbenchmark.utils.LabeledPoint], testData: RDD[MLbenchmark.utils.LabeledPoint], maxIter:Int, chkptIter:Int, optimalVal:Double, lossType:Int) {
		val numFeatures = trainData.take(1)(0).features.size
		var initialWeights = Vectors.dense(new Array[Double](numFeatures))
		val chkptNum = math.floor(maxIter / chkptIter).toInt
		val training = trainData.map(
			point =>{
			var pointLabel = 1
			if (point.label == -1)
				pointLabel = -1
				org.apache.spark.mllib.regression.LabeledPoint(pointLabel, Vectors.sparse(numFeatures, point.features.index, point.features.data))
		}).cache()

		var totalTime: Long = 0
		for(iter <- 1 to chkptNum) {
			val curTime = System.nanoTime()
			//train(RDD<LabeledPoint> input, int numIterations, double stepSize, double regParam, double miniBatchFraction, Vector initialWeights)
			val batchFraction = 1.0
			val model =
				lossType match
				{
					case 0 => SVMWithSGD.train(training, iter * chkptIter, 1.0, 0.002, batchFraction, initialWeights)
					case 1 => RidgeRegressionWithSGD.train(training, iter * chkptIter, 1.0, 0.001, batchFraction, initialWeights)
					case 2 => LassoWithSGD.train(training, iter * chkptIter, 1.0, 0.001, batchFraction, initialWeights)
				}
			totalTime += System.nanoTime() - curTime

			val weightsArray = model.weights.toArray
			initialWeights = Vectors.dense(weightsArray)
			val weightsVector = new DenseVector(model.weights.toArray)
			DebugParamsML.testError(
				weightsVector,
				iter * chkptIter,
				"SGD",
				math.floor(totalTime/1e6).toLong,
				0.001,
				optimalVal,
				lossType,
				trainData,
				testData,
				batchFraction
				)

		}
	}
}