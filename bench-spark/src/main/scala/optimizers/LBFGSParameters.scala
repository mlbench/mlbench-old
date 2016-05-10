package optimizers

import java.io.Serializable

/**
  * Created by amirreza on 10/05/16.
  */
class LBFGSParameters(val iterations: Int = 100,
                      val numCorrections: Int = 10,
                      val convergenceTol: Double = 1E-6,
                      val seed: Int = 13) extends Serializable {
  require(iterations > 0, "iteration must be positive integer")
}
