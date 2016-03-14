import breeze.linalg.DenseVector
import breeze.numerics.log

import scala.math._

/**
  * Created by amirreza on 09/03/16.
  */
object Functions {
  def binaryLogisticLoss(w: DenseVector[Double], xi: DenseVector[Double], yi: Double): Double ={
    //labels must be 1 and -1
    val y:Double = 2 * yi - 1
    return log(1 + exp(-y * w.dot(xi)));
  }

  def l2Regularizer(w: DenseVector[Double]): Double ={
    return 0.5 * w.dot(w);
  }

  def none_reg(w: DenseVector[Double]): Double ={
    return 0;
  }
}
