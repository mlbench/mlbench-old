import breeze.linalg.DenseVector
import breeze.numerics.log

import scala.math._

/**
  * Created by amirreza on 09/03/16.
  */
object Functions {
  def logisticLoss(w: DenseVector[Double], xi: DenseVector[Double], yi: Double): Double ={
    //labels must be 1 and -1
    val y:Double ={
      if (yi == 0)
        -1
      else
        yi
    }
    return log(1 + exp(-y * w.dot(xi)));
  }

  def l2Regularizer(w: DenseVector[Double]): Double ={
    return 0.5 * w.dot(w);
  }
}
