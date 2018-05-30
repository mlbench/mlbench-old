package optimizers

import java.io.Serializable

import breeze.linalg.DenseVector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

/**
  * Created by amirreza on 19/05/16.
  */
class CocoaParameters(var n: Int,
                      var numRounds: Int,
                      var localIterFrac: Double,
                      var lambda: Double,
                      var beta: Double,
                      var gamma: Double,
                      var numParts: Int,
                      var wInit: DenseVector[Double])  extends Serializable  {
  def this(train: RDD[LabeledPoint], test: RDD[LabeledPoint]) {
    this(train.count().toInt,
      200,
      1.0,
      0.01,
      1.0,
      1.0,
      train.partitions.size,
      DenseVector.zeros[Double](train.first().features.size))
  }
  def getLocalIters() = (localIterFrac * n / numParts).toInt

  def getDistOptPar(): distopt.utils.Params ={
    val loss = distopt.utils.OptUtils.hingeLoss _
    return distopt.utils.Params(loss, n, wInit, numRounds, getLocalIters, lambda, beta, gamma)
  }

  override def toString = s"CocoaParameters(n: $n, numRounds: $numRounds, localIters: $getLocalIters, " +
    s"lambda: $lambda, beta: $beta, gamma: $gamma, wInit: $wInit)"
}
