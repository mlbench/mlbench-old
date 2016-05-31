import java.io.Serializable

import breeze.linalg.{DenseVector, Vector}
import l1distopt.utils.DebugParams
import optimizers._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import utils.Functions._
import utils.Utils

/**
  * Created by amirreza on 31/03/16.
  */
object Regression {

  trait Regression extends Serializable {
    def fit(): Vector[Double]
  }

  abstract class LinearRegression(loss: LossFunction,
                                  regularizer: Regularizer)
    extends LinearMethod(loss, regularizer) with Regression {

    override def fit(): Vector[Double] = {
      super.optimize()
    }

    override def predict(w: Vector[Double], test: RDD[org.apache.spark.mllib.linalg.Vector]): RDD[Double] = {
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
  class L1_Lasso_SGD(data: RDD[LabeledPoint],
                     lambda: Double = DEFAULT_LABMDA,
                     params: SGDParameters = new SGDParameters(miniBatchFraction = DEFAULT_BATCH_FRACTION))
    extends LinearRegression(new SquaredLoss, new L1Regularizer(lambda)) with Serializable {
    val optimizer = new SGD(data, loss, regularizer, params)
    require(params.miniBatchFraction < 1.0, "miniBatchFraction must be less than 1. Use GD otherwise.")
  }

  class L1_Lasso_GD(data: RDD[LabeledPoint],
                    lambda: Double = DEFAULT_LABMDA,
                    params: SGDParameters = new SGDParameters(miniBatchFraction = 1.0))
    extends LinearRegression(new SquaredLoss, new L1Regularizer(lambda)) with Serializable {
    val optimizer = new SGD(data, loss, regularizer, params)
    require(params.miniBatchFraction == 1.0, "Use optimizers.SGD for miniBatchFraction less than 1.0")
  }

  class Elastic_ProxCOCOA(train: RDD[LabeledPoint],
                          test: RDD[LabeledPoint],
                          params: ProxCocoaParameters,
                          debug: DebugParams)
    extends LinearRegression(new SquaredLoss, new ElasticNet(params.lambda, params.eta)) {
    def this(train: RDD[LabeledPoint],
             test: RDD[LabeledPoint]) {
      this(train, test, new ProxCocoaParameters(train, test), Utils.defaultDebugProxCocoa(train, test))
    }

    val dataProx = Utils.toProxCocoaTranspose(train)
    val optimizer = new ProxCocoa(dataProx, loss, regularizer, params, debug)
  }

  class L1_Lasso_ProxCocoa(train: RDD[LabeledPoint],
                           test: RDD[LabeledPoint],
                           params: ProxCocoaParameters,
                           debug: DebugParams)
    extends LinearRegression(new SquaredLoss, new ElasticNet(params.lambda, params.eta)) {

    def this(train: RDD[LabeledPoint],
             test: RDD[LabeledPoint]) {
      this(train, test, new ProxCocoaParameters(train ,test, eta = 1.0), Utils.defaultDebugProxCocoa(train, test))
    }

    val dataProx = Utils.toProxCocoaTranspose(train)
    val optimizer = new ProxCocoa(dataProx, loss, regularizer, params, debug)
    require(params.eta == 1.0, "eta must be 1 for L1-regularization")
  }

  class Mllib_Lasso_SGD(data: RDD[LabeledPoint],
                        lambda: Double = DEFAULT_LABMDA,
                        params: SGDParameters = new SGDParameters(miniBatchFraction = DEFAULT_BATCH_FRACTION))
    extends LinearRegression(new SquaredLoss, new L1Regularizer(lambda)) with Serializable {
    val optimizer = new MllibSGD(data, loss, regularizer, params, "Regression")
    require(params.miniBatchFraction < 1.0, "miniBatchFraction must be less than 1. Use GD otherwise.")
  }

  class Mllib_Lasso_GD(data: RDD[LabeledPoint],
                       lambda: Double = DEFAULT_LABMDA,
                       params: SGDParameters = new SGDParameters(miniBatchFraction = 1.0))
    extends LinearRegression(new SquaredLoss, new L1Regularizer(lambda)) with Serializable {
    val optimizer = new MllibSGD(data, loss, regularizer, params, "Regression")
    require(params.miniBatchFraction == 1.0, "Use optimizers.SGD for miniBatchFraction less than 1.0")
  }

}
