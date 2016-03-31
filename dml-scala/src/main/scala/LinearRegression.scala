import Functions._
import Regressions.Regression
import breeze.linalg.DenseVector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

/**
  * Created by amirreza on 09/03/16.
  */
class LinearRegression(loss: LossFunction,
                       regularizer: Regularizer,
                       params: Parameters) extends Regression with Serializable {
  var objectiveValue: Double = -1

  override def fit(data: RDD[LabeledPoint]): DenseVector[Double] = {
    val optimizer: SGD = new SGD(loss, regularizer, params)
    val w: DenseVector[Double] = optimizer.optimize(data)
    objectiveValue = getObjective(w, data)
    return w;
  }

  def getObjective(): Double = {
    if (objectiveValue != -1) {
      return objectiveValue
    } else {
      throw new ClassNotFoundException("You must first fit the model!")
    }
  }

  def getObjective(w: DenseVector[Double], x: RDD[LabeledPoint]): Double = {
    val n: Double = x.count()
    val sum = x.map(p => loss.loss(w, DenseVector(p.features.toArray), p.label)).reduce(_ + _)
    return params.lambda * regularizer.value(w) + (sum / n);
  }

  override def cross_validate(data: RDD[LabeledPoint]): Double = ???
}
