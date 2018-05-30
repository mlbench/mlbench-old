package utils

import java.io.Serializable

import breeze.linalg.{DenseVector, SparseVector, Vector}
import breeze.numerics.log
import org.apache.spark.rdd.RDD

import scala.math._

/**
  * Created by amirreza on 09/03/16.
  */
object Functions {

  type CocoaLabeledPoint = RDD[distopt.utils.LabeledPoint]
  type ProxCocoaLabeledPoint = RDD[l1distopt.utils.LabeledPoint]
  type ProxCocoaDataMatrix = (RDD[(Int, SparseVector[Double])], Vector[Double])

  val DEFAULT_LABMDA = 0.01
  val DEFAULT_BATCH_FRACTION = 0.5
  /*
  * Loss Functions
  * */
  trait LossFunction extends Serializable {
    def loss(w: DenseVector[Double], xi: DenseVector[Double], y: Double): Double

    def subgradient(w: DenseVector[Double], xi: DenseVector[Double], y: Double): DenseVector[Double]
  }

  trait Classifier extends Serializable {
    def classify(z: Double): Double
  }

  class BinaryLogistic extends LossFunction with Classifier {
    def loss(w: DenseVector[Double], xi: DenseVector[Double], y: Double): Double = {
      return log(1.0 + exp(-y * w.dot(xi)));
    }

    def subgradient(w: DenseVector[Double], xi: DenseVector[Double], y: Double): DenseVector[Double] = {
      return xi * (1.0 / (1.0 + exp(-y * w.dot(xi))) - 1.0) * y
    }

    def classify(z: Double): Double = {
      val f = 1.0 / (1 + exp(-1 * z))
      if (f > 0.5) 1 else -1
    }
  }

  class HingeLoss extends LossFunction with Classifier {
    def loss(w: DenseVector[Double], xi: DenseVector[Double], y: Double): Double = {
      return max(0, 1.0 - y * w.dot(xi))
    }

    def subgradient(w: DenseVector[Double], xi: DenseVector[Double], y: Double): DenseVector[Double] = {
      if (y * xi.dot(w) < 1.0)
        -y * xi
      else
        0.0 * xi
    }

    def classify(z: Double): Double = {
      if (z >= 0.0) 1 else -1
    }
  }

  class SquaredLoss extends LossFunction {
    def loss(w: DenseVector[Double], xi: DenseVector[Double], y: Double): Double = {
      val term = w.dot(xi) - y
      return 0.5 * term * term
    }

    def subgradient(w: DenseVector[Double], xi: DenseVector[Double], y: Double): DenseVector[Double] = {
      return (w.dot(xi) - y) * xi
    }
  }

  /*
  *  Regularizers
  * */

  trait Regularizer extends Serializable {
    val lambda: Double
    require(lambda >= 0.0, "Regularizer parameter must be positive")
    def value(w: DenseVector[Double]): Double

    def subgradient(w: DenseVector[Double]): DenseVector[Double]
  }

  class ElasticNet(val lambda: Double, val alpha:Double) extends Regularizer {
    require(alpha >= 0.0, "Elastic net parameter must be positive")
    def value(w: DenseVector[Double]): Double = {
      return alpha * sqrt(w.map(abs(_)).reduceLeft(_ + _)) + (1 - alpha) * 0.5 * w.dot(w);
    }

    def subgradient(w: DenseVector[Double]): DenseVector[Double] = {
      return alpha * w.map(signum(_)) + (1 - alpha) * w;
    }
  }

  class L2Regularizer(val lambda: Double) extends Regularizer {
    def value(w: DenseVector[Double]): Double = {
      return 0.5 * w.dot(w);
    }

    def subgradient(w: DenseVector[Double]): DenseVector[Double] = {
      return w;
    }
  }

  //TODO:Any other more efficient way?
  class L1Regularizer(val lambda: Double) extends Regularizer {
    def value(w: DenseVector[Double]): Double = {
      return sqrt(w.map(abs(_)).reduceLeft(_ + _));
    }

    def subgradient(w: DenseVector[Double]): DenseVector[Double] = {
      return w.map(signum(_));
    }
  }

  class Unregularized(val lambda: Double = 0) extends Regularizer {
    def value(w: DenseVector[Double]): Double = {
      return 0.0;
    }

    def subgradient(w: DenseVector[Double]): DenseVector[Double] = {
      return w * 0.0;
    }
  }
}

