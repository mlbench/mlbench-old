import java.util.Random

import breeze.linalg.DenseVector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

import scala.math._

/**
  * Created by amirreza on 09/03/16.
  */
//Labels must be 0 and 1
class SVM(data: RDD[LabeledPoint]) {
  val rand = new Random(10)
  val ITERATIONS = 50

  def train(): DenseVector[Double] ={
    // Initialize w to a random value
    val D = data.first().features.size
    var w = DenseVector.fill(D){2 * rand.nextDouble - 1}
    println("Initial w: " + w)

    for (i <- 1 to ITERATIONS) {
      val gradient = data.map { p =>
        if((2 * p.label - 1) * DenseVector(p.features.toArray).dot(w) < 1)
          - (2.0 * p.label - 1.0) * DenseVector(p.features.toArray)
        else
          0.0 * DenseVector(p.features.toArray)
      }.reduce(_ + _)
      w -= gradient
    }
    println("Final w: " + w)
    return w;
  }
}
