import breeze.linalg.DenseVector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

import scala.math._

/**
  * Created by amirreza on 09/03/16.
  */

//TODO: Add the regularizer term

//Labels must be 0 and 1
class LogisticRegression(data: RDD[LabeledPoint]) {
  val ITERATIONS = 50

  def train(): DenseVector[Double] ={
    // Initialize w to a random value
    val D = data.first().features.size
    var w = DenseVector.fill(D){0.0}
    println("Initial w: " + w)

    for (i <- 1 to ITERATIONS) {
      val gradient = data.map { p =>
        DenseVector(p.features.toArray) * (1.0 / (1.0 + exp(- (2.0 * p.label - 1.0) * (w.dot(DenseVector(p.features.toArray))))) - 1.0) *
          (2.0 * p.label - 1.0)
      }.reduce(_ + _)
      w -= gradient;
    }
    println("Final w: " + w)
    return w;
  }
}
