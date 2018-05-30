package optimizers

import java.io.Serializable

import breeze.linalg.SparseVector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

/**
  * Created by amirreza on 19/05/16.
  */
class ProxCocoaParameters(var n: Int,
                          var iterations: Int,
                          var localIterFrac: Double,
                          var lambda: Double,
                          var eta: Double,
                          var numFeature: Int,
                          var numParts: Int,
                          var alphaInit: SparseVector[Double])  extends Serializable  {
  def this(train: RDD[LabeledPoint], test: RDD[LabeledPoint], eta: Double = 0.5) {
    this(train.count().toInt,
      100,
      0.9,
      0.1,
      eta,
      train.first().features.size,
      train.partitions.size,
      SparseVector.zeros[Double](train.first().features.size))
  }
  def getLocalIters =  Math.max(1, (localIterFrac * numFeature / numParts).toInt)

  def getL1DistOptPar(): l1distopt.utils.Params = {
    return l1distopt.utils.Params(alphaInit, n, iterations, getLocalIters, lambda, eta)
  }

  override def toString = s"ProxCocoaParameters(n: $n, iterations: $iterations, " +
    s"localIters: $getLocalIters, lambda: $lambda, eta: $eta, alphaInit: $alphaInit)"
}
