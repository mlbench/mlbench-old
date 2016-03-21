import Functions.{HingeLoss, LossFunction, Unregularized, Regularizer}
import breeze.linalg.DenseVector
import breeze.numerics.sqrt
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

/**
  * Created by amirreza on 09/03/16.
  */
class SVM(regularizer: Regularizer = new Unregularized,//No regularizer term by default:
          lambda: Double = 0.0,
          iterations: Int = 100,
          stepSize : Double = 1.0) extends Serializable {
  val loss = new HingeLoss
  var gamma: Double = stepSize

  def train(data: RDD[LabeledPoint]): DenseVector[Double] ={
    // Initialize w to zero
    val d : Int = data.first().features.size
    val n : Double = data.count()
    var w : DenseVector[Double] = DenseVector.fill(d){0.0}

    val eval = new Evaluation(loss = loss, lambda = lambda, regularizer = regularizer)
    for (i <- 1 to iterations) {
      gamma = stepSize / sqrt(iterations)

      val loss_gradient = data.map { p =>
        loss.subgradient(w, DenseVector(p.features.toArray), p.label)
      }.reduce(_ + _)
      val reg_gradient = data.map{ _ => regularizer.subgradient(w)}.reduce(_ + _)

      w -= gamma * (loss_gradient + lambda * reg_gradient)
    }
    return w;
  }

  def getObjective(w: DenseVector[Double], x:  RDD[LabeledPoint]): Double ={
    val n: Double = x.count()
    val sum = x.map(p => loss.loss(w, DenseVector(p.features.toArray), p.label)).reduce(_ + _)
    return lambda * regularizer.value(w) + ( sum / n);
  }

  def classify(w: DenseVector[Double], test: RDD[LabeledPoint]): RDD[(Double,Double)] ={
    val predictions: RDD[(Double,Double)] = test.map(p => (loss.classifier(w.dot(DenseVector(p.features.toArray))), p.label))
    return predictions
  }

  def cross_validate(data: RDD[LabeledPoint]): Double ={
    val Array(d1, d2, d3, d4, d5) = data.randomSplit(Array(0.2, 0.2, 0.2, 0.2, 0.2))

    val train1 = d1.union(d2.union(d3.union(d4)))
    val test1 = d5
    val w1 = this.train(train1)
    val predictions1 = this.classify(w1, test1)
    val error1: Double = predictions1.map( p => if( p._1 != p._2) 1.0 else 0.0).reduce(_ + _) / test1.count()

    val train2 = d2.union(d3.union(d4.union(d5)))
    val test2 = d1
    val w2 = this.train(train2)
    val predictions2 = this.classify(w2, test2)
    val error2:Double = predictions2.map( p => if( p._1 != p._2) 1.0 else 0.0).reduce(_ + _) / test2.count()

    val train3 = d3.union(d4.union(d5.union(d1)))
    val test3 = d2
    val w3 = this.train(train3)
    val predictions3 = this.classify(w3, test3)
    val error3:Double = predictions3.map( p => if( p._1 != p._2) 1.0 else 0.0).reduce(_ + _) / test3.count()

    val train4 = d4.union(d5.union(d1.union(d2)))
    val test4 = d3
    val w4 = this.train(train4)
    val predictions4 = this.classify(w4, test4)
    val error4:Double = predictions4.map( p => if( p._1 != p._2) 1.0 else 0.0).reduce(_ + _) / test4.count()

    val train5 = d5.union(d1.union(d2.union(d3)))
    val test5 = d4
    val w5 = this.train(train5)
    val predictions5 = this.classify(w5, test5)
    val error5:Double = predictions5.map( p => if( p._1 != p._2) 1.0 else 0.0).reduce(_ + _) / test5.count()

    val error:Double = (error1 + error2 + error3 + error4 + error5) / 5.0
    return error
  }
}
