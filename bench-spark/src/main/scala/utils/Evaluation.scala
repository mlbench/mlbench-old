package utils

import java.io.Serializable

import breeze.linalg.DenseVector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import utils.Functions.{HingeLoss, LossFunction, Regularizer, Unregularized}

/**
  * Created by amirreza on 09/03/16.
  */
//TODO: Check boundaries of lambda?
//TODO: Additional getters,setters?
class Evaluation(loss: LossFunction = new HingeLoss,
                 regularizer: Regularizer = new Unregularized,
                 lambda: Double = 0.0) extends Serializable {

  def getObjective(w: DenseVector[Double], x: RDD[LabeledPoint]): Double = {
    val n: Double = x.count()
    val sum = x.map(p => loss.loss(w, DenseVector(p.features.toArray), p.label)).reduce(_ + _)
    return lambda * regularizer.value(w) + (sum / n);
  }

  def error(true_labels: RDD[Double], predictions: RDD[Double]): Double = {
    predictions.zip(true_labels).map(p => if (p._1 != p._2) 1.0 else 0.0).reduce(_ + _) / predictions.count()
  }
}


