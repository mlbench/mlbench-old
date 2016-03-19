import java.io.Serializable

import breeze.linalg.DenseVector
import breeze.numerics.log

import scala.math._

/**
  * Created by amirreza on 09/03/16.
  */
object Functions {

  /*
  * Loss Functions
  * */
  trait LossFunction extends Serializable{
    def loss(w: DenseVector[Double], xi: DenseVector[Double], y: Double): Double

    def subgradient(w: DenseVector[Double], xi: DenseVector[Double], y: Double): DenseVector[Double]
  }



  class BinaryLogistic extends LossFunction{
    def loss(w: DenseVector[Double], xi: DenseVector[Double], y: Double): Double = {
      return log(1.0 + exp(-y * w.dot(xi)));
    }

    def subgradient(w: DenseVector[Double], xi: DenseVector[Double], y: Double): DenseVector[Double] = {
      return xi * (1.0 / (1.0 + exp(-y * w.dot(xi))) - 1.0) * y
    }
  }

  class HingeLoss extends LossFunction {
    def loss(w: DenseVector[Double], xi: DenseVector[Double], y: Double): Double = {
      return max(0, 1.0 - y * w.dot(xi))
    }

    def subgradient(w: DenseVector[Double], xi: DenseVector[Double], y: Double): DenseVector[Double] = {
      if (y * xi.dot(w) < 1.0)
        -y * xi
      else
        0.0 * xi
    }
  }

  class SquaredLoss extends LossFunction{
      def loss(w: DenseVector[Double], xi: DenseVector[Double], y: Double): Double = {
        val  term = w.dot(xi) - y
        return 0.5 * term * term
      }

      def subgradient(w: DenseVector[Double], xi: DenseVector[Double], y: Double): DenseVector[Double] = {
        return (w.dot(xi) - y) * xi
      }
  }

  /*
  *  Regularizers
  * */

  trait Regularizer extends Serializable{
    def value(w: DenseVector[Double]): Double

    def subgradient(w: DenseVector[Double]): DenseVector[Double]
  }

  class L2Regularizer extends Regularizer{
    def value(w: DenseVector[Double]): Double = {
      return 0.5 * w.dot(w);
    }

    def subgradient(w: DenseVector[Double]): DenseVector[Double] = {
      return w;
    }
  }

  class Unregularized extends Regularizer{
    def value(w: DenseVector[Double]): Double = {
      return 0.0;
    }

    def subgradient(w: DenseVector[Double]): DenseVector[Double] = {
      return w * 0.0;
    }
  }
}

