import Functions.{LossFunction, Regularizer}
import breeze.linalg.Vector

/**
  * Created by amirreza on 14/04/16.
  */
abstract class Optimizer[DataType](val loss: LossFunction,
                         val regularizer: Regularizer) extends Serializable {
  def optimize(data: DataType): Vector[Double]
}
