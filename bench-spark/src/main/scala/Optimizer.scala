import Functions.{LossFunction, Regularizer}
import breeze.linalg.DenseVector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

/**
  * Created by amirreza on 14/04/16.
  */
abstract class Optimizer(val loss: LossFunction,
                         val regularizer: Regularizer,
                         val params: SGDParameters) extends Serializable {
  def optimize(data: RDD[LabeledPoint]): DenseVector[Double]
}
