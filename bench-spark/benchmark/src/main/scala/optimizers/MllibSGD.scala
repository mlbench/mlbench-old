package optimizers

import breeze.linalg.{DenseVector, Vector}
import org.apache.spark.mllib.classification.{LogisticRegressionWithSGD, SVMWithSGD}
import org.apache.spark.mllib.optimization.{L1Updater, SimpleUpdater, SquaredL2Updater, Updater}
import org.apache.spark.mllib.regression.{LabeledPoint, LinearRegressionWithSGD}
import org.apache.spark.rdd.RDD
import utils.Functions._

//import scala.tools.cmd.gen.AnyVals.D

/**
  * Created by amirreza on 10/05/16.
  */


class MllibSGD(val data: RDD[LabeledPoint],
               loss: LossFunction,
               regularizer: Regularizer,
               params: SGDParameters,
               ctype: String
              ) extends Optimizer(loss, regularizer) {
  val opt = ctype match {
    case "SVM" => new SVMWithSGD()
    case "LR" => new LogisticRegressionWithSGD()
    case "Regression" => new LinearRegressionWithSGD()
  }

  val reg: Updater = (regularizer: Regularizer) match {
    case _: L1Regularizer => new L1Updater
    case _: L2Regularizer => new SquaredL2Updater
    case _: Unregularized => new SimpleUpdater
  }

  ctype match {
    case "SVM" => opt.asInstanceOf[SVMWithSGD].optimizer.
      setNumIterations(params.iterations).
      setMiniBatchFraction(params.miniBatchFraction).
      setStepSize(params.stepSize).
      setRegParam(regularizer.lambda).
      setUpdater(reg).
      setConvergenceTol(params.convergenceTol)
    case "LR" => opt.asInstanceOf[LogisticRegressionWithSGD].optimizer.
      setNumIterations(params.iterations).
      setMiniBatchFraction(params.miniBatchFraction).
      setStepSize(params.stepSize).
      setRegParam(regularizer.lambda).
      setUpdater(reg).
      setConvergenceTol(params.convergenceTol)
    case "Regression" => opt.asInstanceOf[LinearRegressionWithSGD].optimizer.
      setNumIterations(params.iterations).
      setMiniBatchFraction(params.miniBatchFraction).
      setStepSize(params.stepSize).
      setRegParam(regularizer.lambda).
      setUpdater(reg).
      setConvergenceTol(params.convergenceTol)
  }

  override def optimize(): Vector[Double] = {
    val model = opt.run(data)
    val w = model.weights.toArray
    DenseVector(w)
  }
}
