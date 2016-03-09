import java.io.Serializable

import breeze.linalg.DenseVector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

/**
  * Created by amirreza on 09/03/16.
  */
//TODO: Check boundaries of lambda?
//TODO: Additional getters,setters?


class Evaluation(L: (DenseVector[Double], DenseVector[Double], Double) => Double = Functions.binaryLogisticLoss,
                R: DenseVector[Double] => Double = Functions.l2Regularizer,
                lambda: Double = 0.01) extends Serializable{

  def getObjective(w: DenseVector[Double], x:  RDD[LabeledPoint]): Double ={
    val n = x.collect().length
    val ob = x.map(p => L(w, DenseVector(p.features.toArray), p.label)).reduce(_ + _)
    return lambda * R(w) + (1.0 / n * ob);
  }

}


