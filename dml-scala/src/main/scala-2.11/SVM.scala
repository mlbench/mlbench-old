import breeze.linalg.DenseVector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

/**
  * Created by amirreza on 09/03/16.
  */
class SVM(data: RDD[LabeledPoint]) {
  val ITERATIONS = 50

  def train(): DenseVector[Double] ={
    // Initialize w to zero
    val D = data.first().features.size
    var w = DenseVector.fill(D){0.0}
    println("Initial w: " + w)

    for (i <- 1 to ITERATIONS) {
      val gradient = data.map { p =>
        if( p.label * DenseVector(p.features.toArray).dot(w) < 1)
          - p.label * DenseVector(p.features.toArray)
        else
          0.0 * DenseVector(p.features.toArray)
      }.reduce(_ + _)
      w -= gradient
    }
    println("Final w: " + w)
    return w;
  }
}
