import java.io.Serializable

import Functions._
import breeze.linalg.DenseVector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

/**
  * Created by amirreza on 31/03/16.
  */
object Regressions {

  trait Regression extends Serializable {
    def fit(data: RDD[LabeledPoint]): DenseVector[Double]
  }

  class LinearRegression(loss: LossFunction,
                         regularizer: Regularizer,
                         params: Parameters)
    extends LinearMethod(loss, regularizer, params) with Regression {

    override def fit(data: RDD[LabeledPoint]): DenseVector[Double] = {
      super.optimize(data)
    }

    override def predict(w: DenseVector[Double], test: RDD[LabeledPoint]): RDD[(Double, Double)] = {
      //TODO: Check if label is response values in this data format
      val predictions: RDD[(Double, Double)] = test.map(p => (w.dot(DenseVector(p.features.toArray)), p.label))
      return predictions
    }

    override def error(predictions: RDD[(Double, Double)]): Double = {
      predictions.map(p => (p._2 - p._1) * (p._2 - p._1)).reduce(_ + _) / predictions.count()
    }

  }

  class OrdinaryLeastSquares(params: Parameters = new Parameters)
    extends LinearRegression(new SquaredLoss, new Unregularized, params) with Serializable {
  }

  class RidgeRegression(params: Parameters = new Parameters(lambda = 0.01))
    extends LinearRegression(new SquaredLoss, new L2Regularizer, params) with Serializable {
  }

  class Lasso(params: Parameters = new Parameters(lambda = 0.01))
    extends LinearRegression(new SquaredLoss, new L1Regularizer, params) with Serializable {
  }

}
