package optimizers

import java.io.Serializable

/**
  * Created by amirreza on 16/04/16.
  */
class SGDParameters(val iterations: Int = 20,
                    val miniBatchFraction: Double = 1.0,
                    val stepSize: Double = 0.001,
                    val seed: Int = 13) extends Serializable {
  require(iterations > 0, "iteration must be positive integer")
  require(miniBatchFraction > 0 && miniBatchFraction <= 1.0, "miniBatchFraction must be between 0 and 1")
}
