import java.io.Serializable

import breeze.linalg.{DenseVector, Vector}
import optimizers.Optimizer
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import utils.Functions.{LossFunction, Regularizer}

/**
  * Created by amirreza on 09/03/16.
  */
abstract class LinearMethod[DataType](val loss: LossFunction,
                            val regularizer: Regularizer) extends Serializable {
  val optimizer: Optimizer[DataType]

  def optimize(data: DataType): Vector[Double] = {
    val w: Vector[Double] = optimizer.optimize(data)
    return w;
  }

  def predict(w: Vector[Double], test: RDD[org.apache.spark.mllib.linalg.Vector]): RDD[Double]

  def error(trueLabels: RDD[Double], predictions: RDD[Double]): Double

  def testError(w: Vector[Double], test: RDD[org.apache.spark.mllib.linalg.Vector], trueLabels: RDD[Double]): Double = {
    val predictions = predict(w, test)
    val err = error(trueLabels, predictions)
    return err
  }

  def getObjective(w: DenseVector[Double], x: RDD[LabeledPoint]): Double = {
    val n: Double = x.count()
    val sum = x.map(p => loss.loss(w, DenseVector(p.features.toArray), p.label)).reduce(_ + _)
    return regularizer.lambda * regularizer.value(w) + (sum / n);
  }
}
