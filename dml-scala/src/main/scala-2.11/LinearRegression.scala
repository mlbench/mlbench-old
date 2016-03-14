import breeze.linalg.DenseVector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

/**
  * Created by amirreza on 09/03/16.
  */
class LinearRegression(data: RDD[LabeledPoint]) {
  val ITERATIONS = 50

  def train(): DenseVector[Double] ={
    // Initialize w to a random value
    val D = data.first().features.size
    var w = DenseVector.fill(D){0.0}
    println("Initial w: " + w)

    for (i <- 1 to ITERATIONS) {
      val gradient = data.map { p =>
        (w.dot(DenseVector(p.features.toArray)) - p.label) * DenseVector(p.features.toArray)
      }.reduce(_ + _)
      w -= gradient
    }
    println("Final w: " + w)
    return w;
  }
}
