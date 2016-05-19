package optimizers

import java.io.Serializable

/**
  * Created by amirreza on 10/05/16.
  */
class LBFGSParameters(var iterations: Int = 100,
                      var numCorrections: Int = 10,
                      var convergenceTol: Double = 1E-6,
                      var seed: Int = 13) extends Serializable {
  require(iterations > 0, "iteration must be positive integer")

  override def toString = s"LBFGSParameters(iterations: $iterations, numCorrections: $numCorrections, " +
    s"convergenceTol: $convergenceTol, seed: $seed)"
}
