import Functions.Unregularized
import breeze.linalg.DenseVector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

/**
  * Created by amirreza on 09/03/16.
  */
class LinearRegression(data: RDD[LabeledPoint],
                       //No regularizer term by default
                       reg_gradient: Functions.Regularizer = new Unregularized,
                       lambda: Double = 0.01) {
  val ITERATIONS = 50

  def train(): DenseVector[Double] ={
    // Initialize w to zero
    val D = data.first().features.size
    var w = DenseVector.fill(D){0.0}

    for (i <- 1 to ITERATIONS) {
      val gradient = data.map { p =>
        (w.dot(DenseVector(p.features.toArray)) - p.label) * DenseVector(p.features.toArray)
      }.reduce(_ + _)
      w -= (gradient + lambda * reg_gradient.subgradient(w))
    }
    println("Final w: " + w)
    return w;
  }
}
