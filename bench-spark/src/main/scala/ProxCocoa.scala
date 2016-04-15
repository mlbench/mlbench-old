import Functions.{LossFunction, ProxCocoaDataMatrix, Regularizer}
import breeze.linalg.Vector
import l1distopt.solvers.ProxCoCoAp
import l1distopt.utils.{DebugParams, Params}

/**
  * Created by amirreza on 15/04/16.
  */


class ProxCocoa(loss: LossFunction,
                regularizer: Regularizer,
                params: Params,
                debug: DebugParams) extends Optimizer[ProxCocoaDataMatrix] (loss, regularizer){
  override def optimize(data: ProxCocoaDataMatrix): Vector[Double] = {
    val finalAlphaCoCoA = ProxCoCoAp.runProxCoCoAp(data._1, data._2, params, debug)
    return finalAlphaCoCoA
  }
}
