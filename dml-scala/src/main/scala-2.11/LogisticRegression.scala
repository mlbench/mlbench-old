import breeze.linalg.DenseVector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import java.util.Random

import scala.math._

/**
  * Created by amirreza on 09/03/16.
  */

//TODO: Appropriate choice of initial w?
//TODO: Appropriate choice of ITERATIONS
class LogisticRegression(data: RDD[LabeledPoint]) {
  val rand = new Random(10)
  val ITERATIONS = 50

    def train(): DenseVector[Double] ={
      // Initialize w to a random value
      val D = data.first().features.size
      var w = DenseVector.fill(D){2 * rand.nextDouble - 1}
      println("Initial w: " + w)

      for (i <- 1 to ITERATIONS) {
        val gradient = data.map { p =>
          DenseVector(p.features.toArray) * (1 / (1 + exp(-p.label * (w.dot(DenseVector(p.features.toArray))))) - 1) * p.label
        }.reduce(_ + _)
        w -= gradient
      }
      println("Final w: " + w)
      return w;
    }
}
