package MLbenchmark.solvers

import org.apache.spark.SparkContext
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.mllib.regression.{RidgeRegressionWithSGD, LabeledPoint, RidgeRegressionModel}
import org.apache.spark.mllib.optimization.{LBFGS, LeastSquaresGradient, SquaredL2Updater}
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.util.MLUtils

object Lbfgs{
    def run_LBFGS(train_data: RDD[LabeledPoint], test_data: RDD[LabeledPoint]){
      val numFeatures = train_data.take(1)(0).features.size

      // Run training algorithm to build the model
      val numCorrections = 10
      val convergenceTol = 1e-4
      val maxNumIterations = 20
      val regParam = 0.1
      val initialWeightsWithIntercept = Vectors.dense(new Array[Double](numFeatures + 1))

      val training = train_data.map(point => (point.label, MLUtils.appendBias(point.features)))
      val (weightsWithIntercept, loss) = LBFGS.runLBFGS(
        training,
        new LeastSquaresGradient(),
        new SquaredL2Updater(),
        numCorrections,
        convergenceTol,
        maxNumIterations,
        regParam,
        initialWeightsWithIntercept)

      val model = new RidgeRegressionModel(
        Vectors.dense(weightsWithIntercept.toArray.slice(0, weightsWithIntercept.size - 1)),
        weightsWithIntercept(weightsWithIntercept.size - 1))

      val valAndPreds = test_data.map{ 
      point => 
      val prediction = model.predict(point.features)
      (point.label, prediction)
      }
      val MSE = valAndPreds.map{case(v,p) => math.pow((v-p),2)}.mean()
      println("LBFGS training Mean Squared Error = " + MSE)
  }
}