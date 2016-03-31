import java.io.Serializable

import Functions._
import breeze.linalg.DenseVector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

/**
  * Created by amirreza on 09/03/16.
  */
abstract class LinearMethod(loss: LossFunction,
                            regularizer: Regularizer,
                            params: Parameters) extends Serializable {
  var objectiveValue: Option[Double] = None

  def optimize(data: RDD[LabeledPoint]): DenseVector[Double] = {
    val optimizer: SGD = new SGD(loss, regularizer, params)
    val w: DenseVector[Double] = optimizer.optimize(data)
    objectiveValue = Some(getObjective(w, data))
    return w;
  }

  def predict(w: DenseVector[Double], test: RDD[LabeledPoint]): RDD[(Double, Double)]

  def error(data: RDD[(Double, Double)]): Double

  def cvError(data: RDD[LabeledPoint]): Double = {
    val Array(d1, d2, d3, d4, d5) = data.randomSplit(Array(0.2, 0.2, 0.2, 0.2, 0.2))

    val train1 = d1.union(d2.union(d3.union(d4)))
    val test1 = d5
    val w1 = this.optimize(train1)
    val predictions1 = this.predict(w1, test1)
    val error1: Double = this.error(predictions1)

    val train2 = d2.union(d3.union(d4.union(d5)))
    val test2 = d1
    val w2 = this.optimize(train2)
    val predictions2 = this.predict(w2, test2)
    val error2: Double = this.error(predictions2)

    val train3 = d3.union(d4.union(d5.union(d1)))
    val test3 = d2
    val w3 = this.optimize(train3)
    val predictions3 = this.predict(w3, test3)
    val error3: Double = this.error(predictions3)

    val train4 = d4.union(d5.union(d1.union(d2)))
    val test4 = d3
    val w4 = this.optimize(train4)
    val predictions4 = this.predict(w4, test4)
    val error4: Double = this.error(predictions4)

    val train5 = d5.union(d1.union(d2.union(d3)))
    val test5 = d4
    val w5 = this.optimize(train5)
    val predictions5 = this.predict(w5, test5)
    val error5: Double = this.error(predictions5)

    val error: Double = (error1 + error2 + error3 + error4 + error5) / 5.0
    return error
  }

  def getObjective(): Double = {
    objectiveValue match {
      case None => throw new ClassNotFoundException("You must first train the classifier")
      case Some(ob) => ob
    }
  }

  def getObjective(w: DenseVector[Double], x: RDD[LabeledPoint]): Double = {
    val n: Double = x.count()
    val sum = x.map(p => loss.loss(w, DenseVector(p.features.toArray), p.label)).reduce(_ + _)
    return params.lambda * regularizer.value(w) + (sum / n);
  }
}
