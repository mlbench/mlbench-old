import Functions.{HingeLoss, LossFunction, Unregularized, Regularizer}
import breeze.linalg.DenseVector
import breeze.numerics.sqrt
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

/**
  * Created by amirreza on 09/03/16.
  */
class SVM(data: RDD[LabeledPoint],
          //No regularizer term by default:
          regularizer: Regularizer = new Unregularized,
          lambda: Double = 0.01,
          iterations: Int = 100) {

  val stepSize: Double = 1.0
  var gamma: Double = stepSize

  def train(): DenseVector[Double] ={
    // Initialize w to zero
    val d : Int = data.first().features.size
    val n : Double = data.collect().length
    var w : DenseVector[Double] = DenseVector.fill(d){0.0}
    val loss:LossFunction = new HingeLoss

    for (i <- 1 to iterations) {
      gamma = stepSize / sqrt(iterations)
      val gradient = data.map { p =>
        loss.subgradient(w, DenseVector(p.features.toArray), p.label)
      }.reduce(_ + _)
      w -= gamma * (gradient + lambda * regularizer.subgradient(w) * n)
    }
    println("SVM w: " + w)
    return w;
  }
}
