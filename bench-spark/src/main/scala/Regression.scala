import java.io.Serializable

import Functions._
import breeze.linalg.DenseVector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

/**
  * Created by amirreza on 31/03/16.
  */
object Regression {

  trait Regression extends Serializable {
    def fit(data: RDD[LabeledPoint]): DenseVector[Double]
  }

  abstract class LinearRegression(loss: LossFunction,
                                  regularizer: Regularizer)
    extends LinearMethod(loss, regularizer) with Regression {

    override def fit(data: RDD[LabeledPoint]): DenseVector[Double] = {
      super.optimize(data)
    }

    override def predict(w: DenseVector[Double], test: RDD[org.apache.spark.mllib.linalg.Vector]): RDD[Double] = {
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
    extends LinearRegression(new SquaredLoss, new L1Regularizer(lambda)) with Serializable {
    val optimizer: Optimizer = new SGD(loss, regularizer, params)
    require(params.miniBatchFraction < 1.0, "miniBatchFraction must be less than 1. Use GD otherwise.")
  }

  class L1_Lasso_GD(lambda: Double = 0.1,
                    params: SGDParameters = new SGDParameters(miniBatchFraction = 1.0))
    extends LinearRegression(new SquaredLoss, new L1Regularizer(lambda)) with Serializable {
    val optimizer: Optimizer = new SGD(loss, regularizer, params)
    require(params.miniBatchFraction == 1.0, "Use SGD for miniBatchFraction less than 1.0")
  }

}
