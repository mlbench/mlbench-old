package optimizers

import java.io.Serializable

/**
  * Created by amirreza on 16/04/16.
  */
class SGDParameters(var iterations: Int = 100,
                    var miniBatchFraction: Double = 0.5,
                    var stepSize: Double = 0.001,
                    var seed: Int = 13,
                    var convergenceTol: Double = 0.001) extends Serializable {
  require(iterations > 0, "iteration must be positive integer")
  require(miniBatchFraction > 0 && miniBatchFraction <= 1.0, "miniBatchFraction must be between 0 and 1")
  require(convergenceTol > 0, "convergenceTol must be greater than 0")
  override def toString = s"SGDParameters(iterations: $iterations, minibatchFraction: $miniBatchFraction, " +
    s"stepSize: $stepSize, seed: $seed, convergenceTol: $convergenceTol)"
}
