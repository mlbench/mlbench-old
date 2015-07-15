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
	def run_SGD(trainData: RDD[MLbenchmark.utils.LabeledPoint], testData: RDD[MLbenchmark.utils.LabeledPoint], maxIter:Int, chkptIter:Int,debugML:DebugParamsML)
	{
		val numFeatures = trainData.take(1)(0).features.size
		var initialWeights = Vectors.dense(new Array[Double](numFeatures))
		val chkptNum = math.floor(maxIter/chkptIter).toInt
		val training = trainData.map(point => org.apache.spark.mllib.regression.LabeledPoint(point.label, Vectors.sparse(numFeatures, point.features.index, point.features.data))).cache()
		var totalTime: Long = 0
		for(iter <- 1 to chkptNum) {
			val curTime = System.nanoTime()
			//train(RDD<LabeledPoint> input, int numIterations, double stepSize, double regParam, double miniBatchFraction, Vector initialWeights)
			val model = RidgeRegressionWithSGD.train(training, iter * chkptIter, 1.0, 0.001, 1.0 , initialWeights)
			totalTime += System.nanoTime() - curTime

			val weightsArray = model.weights.toArray
			initialWeights = Vectors.dense(weightsArray)
			val weightsVector = new DenseVector(model.weights.toArray)
			debugML.testError(weightsVector,iter * chkptIter,"SGD", math.floor(totalTime/1e6).toLong)
		}
	}
}