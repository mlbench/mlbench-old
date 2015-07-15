package MLbenchmark.solvers

import org.apache.spark.SparkContext
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.regression.{RidgeRegressionWithSGD, RidgeRegressionModel}
import org.apache.spark.mllib.optimization.{LBFGS, LeastSquaresGradient, SquaredL2Updater}
import org.apache.spark.rdd.RDD
import MLbenchmark.utils._
import breeze.linalg.DenseVector

object Lbfgs{
    def run_LBFGS(trainData: RDD[LabeledPoint], testData: RDD[LabeledPoint],maxNumIterations:Int, chkptIter:Int,debugML:DebugParamsML){

      val numFeatures = trainData.take(1)(0).features.size

      // Run training algorithm to build the model
      val numCorrections = 20
      val convergenceTol = 1e-4
      val regParam = 1.0
      var initialWeights = Vectors.dense(new Array[Double](numFeatures))
      val training = trainData.map(point => (point.label, Vectors.sparse(numFeatures, point.features.index, point.features.data)))

      val chkptNum = math.floor(maxNumIterations/chkptIter).toInt
      var totalTime: Long = 0
      for(iter<-1 to chkptNum) {
        val curTime = System.nanoTime()
        val (weights, loss) = LBFGS.runLBFGS(
          training,
          new LeastSquaresGradient(),
          new SquaredL2Updater(),
          numCorrections,
          convergenceTol,
          chkptIter,
          0.001,
          initialWeights)
          totalTime += System.nanoTime() - curTime

          val weightsVector = new DenseVector(weights.toArray)
          initialWeights = Vectors.dense(weights.toArray)
          debugML.testError(weightsVector, iter*chkptIter,"Lbfgs", math.floor(totalTime/1e6).toLong)
      }
  }
}