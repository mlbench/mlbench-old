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

    def cross_validate(data: RDD[LabeledPoint]): Double

    def getObjective(): Double

    def getObjective(w: DenseVector[Double], x: RDD[LabeledPoint]): Double
  }

  class OrdinaryLeastSquares(params: Parameters = new Parameters) extends Regression with Serializable {
    val ordinary = new LinearRegression(new SquaredLoss, new Unregularized, params)

    override def fit(data: RDD[LabeledPoint]): DenseVector[Double] =
      ordinary.fit(data)

    override def cross_validate(data: RDD[LabeledPoint]): Double =
      ordinary.cross_validate(data)

    override def getObjective(): Double =
      ordinary.getObjective()

    override def getObjective(w: DenseVector[Double], x: RDD[LabeledPoint]): Double =
      ordinary.getObjective(w, x)
  }

  class RidgeRegression(params: Parameters = new Parameters(lambda = 0.01)) extends Regression with Serializable {
    val ridge = new LinearRegression(new SquaredLoss, new L2Regularizer, params)

    override def fit(data: RDD[LabeledPoint]): DenseVector[Double] =
      ridge.fit(data)

    override def cross_validate(data: RDD[LabeledPoint]): Double =
      ridge.cross_validate(data)

    override def getObjective(): Double =
      ridge.getObjective()

    override def getObjective(w: DenseVector[Double], x: RDD[LabeledPoint]): Double =
      ridge.getObjective(w, x)
  }

  class Lasso(params: Parameters = new Parameters(lambda = 0.01)) extends Regression with Serializable {
    val lasso = new LinearRegression(new SquaredLoss, new L1Regularizer, params)

    override def fit(data: RDD[LabeledPoint]): DenseVector[Double] =
      lasso.fit(data)

    override def cross_validate(data: RDD[LabeledPoint]): Double =
      lasso.cross_validate(data)

    override def getObjective(): Double =
      lasso.getObjective()

    override def getObjective(w: DenseVector[Double], x: RDD[LabeledPoint]): Double =
      lasso.getObjective(w, x)
  }

}
