import Functions.{LossFunction, Regularizer, SquaredLoss, Unregularized}
import breeze.linalg.DenseVector
import breeze.numerics._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

/**
  * Created by amirreza on 09/03/16.
  */
class LinearRegression(data: RDD[LabeledPoint],
                       //No regularizer term by default
                       regularizer: Regularizer = new Unregularized,
                       lambda: Double = 0.0,
                       iterations: Int = 100,
                       stepSize : Double = 1.0) extends Serializable{
  var gamma:Double = stepSize

  def train(): DenseVector[Double] ={
    // Initialize w to zero
    val d : Int = data.first().features.size
    val n : Double = data.count()
    var w : DenseVector[Double] = DenseVector.fill(d){0.0}
    val loss: LossFunction = new SquaredLoss

    for (i <- 1 to iterations) {
      gamma = stepSize / sqrt(iterations)

      val loss_gradient = data.map { p =>
        loss.subgradient(w, DenseVector(p.features.toArray), p.label)
      }.reduce(_ + _)
      val reg_gradient = data.map(_ => regularizer.subgradient(w)).reduce(_ + _)

      w -= gamma * (loss_gradient + lambda *  reg_gradient)
    }
    println("Regression w: " + w)
    return w;
  }
}
