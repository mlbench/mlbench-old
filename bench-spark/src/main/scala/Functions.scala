import java.io.Serializable

import breeze.linalg.DenseVector
import breeze.math.MutablizingAdaptor.Lambda2
import breeze.numerics.log

import scala.math._

/**
  * Created by amirreza on 09/03/16.
  */
object Functions {

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

    def value(w: DenseVector[Double]): Double

    def subgradient(w: DenseVector[Double]): DenseVector[Double]
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

  class Parameters(val iterations: Int = 100,
                   val miniBatchFraction: Double = 1.0,
                   val stepSize: Double = 1.0,
                   val seed: Int = 13) extends Serializable {

  }

}

