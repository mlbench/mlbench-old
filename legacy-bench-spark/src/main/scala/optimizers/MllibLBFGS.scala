package optimizers

import breeze.linalg.{DenseVector, Vector}
import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.optimization.{L1Updater, SimpleUpdater, SquaredL2Updater, Updater}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import utils.Functions._

/**
  * Created by amirreza on 10/05/16.
  */
class MllibLBFGS(val data: RDD[LabeledPoint],
                 loss: LossFunction,
                 regularizer: Regularizer,
                 params: LBFGSParameters
                ) extends Optimizer(loss, regularizer) {

  val opt = new LogisticRegressionWithLBFGS

  val reg: Updater = (regularizer: Regularizer) match {
    case _: L1Regularizer => new L1Updater
    case _: L2Regularizer => new SquaredL2Updater
    case _: Unregularized => new SimpleUpdater
  }

  opt.optimizer.
    setNumIterations(params.iterations).
    setConvergenceTol(params.convergenceTol).
    setNumCorrections(params.numCorrections).
    setRegParam(regularizer.lambda).
    setUpdater(reg)

  override def optimize(): Vector[Double] = {
    val model = opt.run(data)
    val w = model.weights.toArray
    return DenseVector(w)
  }
}
