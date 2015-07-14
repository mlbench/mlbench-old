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
    def run_LBFGS(trainData: RDD[LabeledPoint], testData: RDD[LabeledPoint],maxNumIterations:Int){
      val numFeatures = trainData.take(1)(0).features.size

      // Run training algorithm to build the model
      val numCorrections = 20
      val convergenceTol = 1e-4
      val regParam = 1.0
      val initialWeights = Vectors.dense(new Array[Double](numFeatures))

      val training = trainData.map(point => (point.label, Vectors.sparse(numFeatures, point.features.index, point.features.data)))
      val (weights, loss) = LBFGS.runLBFGS(
        training,
        new LeastSquaresGradient(),
        new SquaredL2Updater(),
        numCorrections,
        convergenceTol,
        maxNumIterations,
        regParam,
        initialWeights)

      val weightsVector = new DenseVector(weights.toArray)
      val testError = OptUtils.computeClassificationError(testData, weightsVector)
      println("test Error:" + testError)
      val objectFunctionCheck = OptUtils.computeObjective(weightsVector)
      println("object function" + objectFunctionCheck)
      val model = new RidgeRegressionModel(
        Vectors.dense(weights.toArray),
        0)

      
      val valAndPreds = testData.map{ 
      point => 
      val prediction = model.predict(Vectors.sparse(numFeatures,point.features.index, point.features.data))
      (point.label, prediction)
      }
      val MSE = valAndPreds.map{case(v,p) => math.pow((v-p),2)}.mean()
      println("LBFGS training Mean Squared Error = " + MSE)
  }
}