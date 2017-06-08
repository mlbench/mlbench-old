package optimizers

import utils.Functions.{LossFunction, Regularizer}
import breeze.linalg.Vector

/**
  * Created by amirreza on 14/04/16.
  */
abstract class Optimizer(val loss: LossFunction,
                         val regularizer: Regularizer) extends Serializable {

  val data: Any

  //Data Type depends on the optimizer
  def optimize(): Vector[Double]
}
