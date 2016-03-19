import java.io.Serializable

import Functions.{Unregularized, Regularizer, HingeLoss, LossFunction}
import breeze.linalg.DenseVector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

/**
  * Created by amirreza on 09/03/16.
  */
//TODO: Check boundaries of lambda?
//TODO: Additional getters,setters?
class Evaluation(loss: LossFunction = new HingeLoss,
                 regularizer: Regularizer = new Unregularized,
                 lambda: Double = 0.01) extends Serializable{

  def getObjective(w: DenseVector[Double], x:  RDD[LabeledPoint]): Double ={
    val n = x.collect().length
    val sum = x.map(p => loss.loss(w, DenseVector(p.features.toArray), p.label)).reduce(_ + _)
    return lambda * regularizer.value(w) + ( sum / n);
  }

}


