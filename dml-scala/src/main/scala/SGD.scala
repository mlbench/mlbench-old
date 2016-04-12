import Functions._
import breeze.linalg.DenseVector
import breeze.numerics.sqrt
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

import scala.util.Random


/**
  * Created by amirreza on 09/03/16.
  */

abstract class Optimizer(val loss: LossFunction,
                         val regularizer: Regularizer,
                         val params: Parameters) extends Serializable{
  def optimize(data:RDD[LabeledPoint]): DenseVector[Double]
}

class SGD(loss: LossFunction,
          regularizer: Regularizer = new Unregularized, //No regularizer term by default:
          params: Parameters) extends Optimizer(loss, regularizer, params){


  override def optimize(data: RDD[LabeledPoint]): DenseVector[Double] = {
    val d: Int = data.first().features.size //feature dimension
    val n: Double = data.count() //dataset size

    var gamma: Double = params.stepSize
    var w: DenseVector[Double] = DenseVector.fill(d) {
      0.0
    } //Initial weight vector

    //TODO: Isn't this inefficient ??!!
    val dataArr = data.mapPartitions(x => Iterator(x.toArray))
    for (i <- 1 to params.iterations) {
      gamma = params.stepSize / sqrt(i)
      val loss_gradient = dataArr.mapPartitions(partitionUpdate(_, w, params.miniBatchFraction, params.seed)).reduce(_ + _)
      val reg_gradient: DenseVector[Double] = regularizer.subgradient(w) * n
      w -= gamma * (loss_gradient + regularizer.lambda * reg_gradient)
    }

    return w;
  }

  private def partitionUpdate(localData: Iterator[Array[LabeledPoint]],
                              w: DenseVector[Double],
                              fraction: Double,
                              seed: Int): Iterator[DenseVector[Double]] = {
    val array: Array[LabeledPoint] = localData.next() //Get the array
    val n: Int = array.length //local dataset size
    val subSetSize: Int = (n * fraction).toInt //size of randomely selected samples
    require(subSetSize > 0, "fraction is too small: " + fraction)
    //Randomely sample local dataset
    val r = new Random(seed)
    val subSet = r.shuffle(array.toList).take(subSetSize) //TODO: Isn't this inefficient?

    val res = subSet.map(p =>
      loss.subgradient(w, DenseVector(p.features.toArray), p.label)).reduce(_ + _)
    return Iterator(res)
  }
}
