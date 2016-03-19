import Functions._
import breeze.linalg.{DenseVector, min}
import breeze.numerics.sqrt
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

/**
  * Created by amirreza on 09/03/16.
  */
class LogisticRegression(data: RDD[LabeledPoint],
                         //No regularizer term by default:
                         regularizer: Regularizer = new Unregularized,
                         lambda: Double = 0.0,
                         iterations: Int = 100,
                         stepSize : Double = 1.0) {
  var gamma:Double = stepSize

  def train(): DenseVector[Double] ={
    // Initialize w to zero
    val d : Int = data.first().features.size
    val n : Double = data.count()
    var w : DenseVector[Double] = DenseVector.fill(d){0.0}
    val loss:LossFunction = new BinaryLogistic

    val eval = new Evaluation(loss, regularizer, lambda)
    for (i <- 1 to iterations) {
      gamma = stepSize / sqrt(iterations)
      val gradient = data.map { p =>
        loss.subgradient(w, DenseVector(p.features.toArray), p.label)
      }.reduce(_ + _)
      w -= gamma * (gradient + lambda * regularizer.subgradient(w) * n)
    }
    println("Logistic w: " + w)
    return w;
  }
}
