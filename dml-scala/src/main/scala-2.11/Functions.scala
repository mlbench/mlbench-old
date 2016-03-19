import java.io.Serializable

import breeze.linalg.DenseVector
import breeze.numerics.log

import scala.math._

/**
  * Created by amirreza on 09/03/16.
  */
object Functions {

  trait LossFunction {
    def loss(w: DenseVector[Double], xi: DenseVector[Double], y: Double): Double

    def subgradient(w: DenseVector[Double], xi: DenseVector[Double], y: Double): DenseVector[Double]
  }

  trait Regularizer {
    def value(w: DenseVector[Double]): Double

    def subgradient(w: DenseVector[Double]): DenseVector[Double]
  }

  class BinaryLogistic extends LossFunction with Serializable  {
    def loss(w: DenseVector[Double], xi: DenseVector[Double], y: Double): Double = {
      return log(1.0 + exp(-y * w.dot(xi)));
    }

    def subgradient(w: DenseVector[Double], xi: DenseVector[Double], y: Double): DenseVector[Double] = {
      return xi * (1.0 / (1.0 + exp(-y * w.dot(xi))) - 1.0) * y
    }
  }

  class HingeLoss extends LossFunction with Serializable {
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

  class L2Regularizer extends Regularizer with Serializable  {
    def value(w: DenseVector[Double]): Double = {
      return 0.5 * w.dot(w);
    }

    def subgradient(w: DenseVector[Double]): DenseVector[Double] = {
      return w;
    }
  }

  class Unregularized extends Regularizer with Serializable  {
    def value(w: DenseVector[Double]): Double = {
      return 0.0;
    }

    def subgradient(w: DenseVector[Double]): DenseVector[Double] = {
      return w * 0.0;
    }
  }

}

