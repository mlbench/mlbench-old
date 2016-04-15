import java.io.Serializable

import Classification.LinearClassifier
import Functions._
import breeze.linalg.{DenseVector, Vector}
import l1distopt.utils.{DebugParams, Params}
import org.apache.spark.rdd.RDD

/**
  * Created by amirreza on 31/03/16.
  */
object Regression {

  trait Regression[DataType] extends Serializable {
    def fit(data: DataType): Vector[Double]
  }

  abstract class LinearRegression[DataType](loss: LossFunction,
                                            regularizer: Regularizer)
    extends LinearMethod[DataType](loss, regularizer) with Regression[DataType] {

    override def fit(data: DataType): Vector[Double] = {
      super.optimize(data)
    }

    override def predict(w: Vector[Double], test: RDD[org.apache.spark.mllib.linalg.Vector]): RDD[Double] = {
      //TODO: Check if label is response values in this data format
      //TODO: Isn't converting to DenseVector costly?
      val predictions: RDD[Double] = test.map(p => w.dot(DenseVector(p.toArray)))
      return predictions
    }

    //Mean squared error
    override def error(trueLabels: RDD[Double], predictions: RDD[Double]): Double = {
      predictions.zip(trueLabels).map(p => (p._2 - p._1) * (p._2 - p._1)).reduce(_ + _) / predictions.count()
    }

  }

  /*
   Tasks L1:
  */
  class L1_Lasso_SGD(lambda: Double = 0.1,
                     params: SGDParameters = new SGDParameters(miniBatchFraction = 0.5))
    extends LinearRegression[SGDDataMatrix](new SquaredLoss, new L1Regularizer(lambda)) with Serializable {
    val optimizer = new SGD(loss, regularizer, params)
    require(params.miniBatchFraction < 1.0, "miniBatchFraction must be less than 1. Use GD otherwise.")
  }

  class L1_Lasso_GD(lambda: Double = 0.1,
                    params: SGDParameters = new SGDParameters(miniBatchFraction = 1.0))
    extends LinearRegression[SGDDataMatrix](new SquaredLoss, new L1Regularizer(lambda)) with Serializable {
    val optimizer = new SGD(loss, regularizer, params)
    require(params.miniBatchFraction == 1.0, "Use SGD for miniBatchFraction less than 1.0")
  }

  class Elastic_ProxCOCOA(lambda: Double = 0.01,
                          alpha: Double = 0.01,
                          params: Params,
                          debug: DebugParams)
    extends LinearRegression[ProxCocoaDataMatrix](new SquaredLoss, new ElasticNet(lambda, alpha)) {
    val optimizer = new ProxCocoa(loss, regularizer, params, debug)

  }

}
