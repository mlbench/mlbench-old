import breeze.linalg.DenseVector
import breeze.numerics.log

import scala.math._

/**
  * Created by amirreza on 09/03/16.
  */
object Functions {
  def binaryLogisticLoss(w: DenseVector[Double], xi: DenseVector[Double], y: Double): Double ={
    return log(1 + exp(-y * w.dot(xi)));
  }

  def l2Regularizer(w: DenseVector[Double]): Double ={
    return 0.5 * w.dot(w);
  }

  def hingeLoss(w: DenseVector[Double], xi:DenseVector[Double], y: Double): Double ={
    return max(0, 1 - y * w.dot(xi))
  }

  def none_reg(w: DenseVector[Double]): Double ={
    return 0;
  }
}
