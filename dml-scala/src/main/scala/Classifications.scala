import java.io.Serializable

import Functions._
import breeze.linalg.DenseVector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

/**
  * Created by amirreza on 31/03/16.
  */
object Classifications {


  trait Classification extends Serializable {
    def train(data: RDD[LabeledPoint]): DenseVector[Double]

    def classify(w: DenseVector[Double], test: RDD[LabeledPoint]): RDD[(Double, Double)]

    def cross_validate(data: RDD[LabeledPoint]): Double

    def getObjective(): Double

    def getObjective(w: DenseVector[Double], x: RDD[LabeledPoint]): Double
  }

  class SVM(regularizer: Regularizer = new Unregularized, //No regularizer term by default:
            params: Parameters = new Parameters) extends Classification with Serializable {
    val svmClassifier = new LinearClassifier(new HingeLoss, regularizer, params)

    override def train(data: RDD[LabeledPoint]): DenseVector[Double] =
      svmClassifier.train(data)

    override def cross_validate(data: RDD[LabeledPoint]): Double =
      svmClassifier.cross_validate(data)

    override def getObjective(): Double =
      svmClassifier.getObjective()

    override def getObjective(w: DenseVector[Double], x: RDD[LabeledPoint]): Double =
      svmClassifier.getObjective(w, x)

    override def classify(w: DenseVector[Double], test: RDD[LabeledPoint]): RDD[(Double, Double)] =
      svmClassifier.classify(w, test)
  }

  class LogisticRegression(regularizer: Regularizer = new Unregularized, //No regularizer term by default:
                           params: Parameters = new Parameters) extends Classification with Serializable {
    val lrClassifier = new LinearClassifier(new BinaryLogistic, regularizer, params)

    override def train(data: RDD[LabeledPoint]): DenseVector[Double] =
      lrClassifier.train(data)

    override def cross_validate(data: RDD[LabeledPoint]): Double =
      lrClassifier.cross_validate(data)

    override def classify(w: DenseVector[Double], test: RDD[LabeledPoint]): RDD[(Double, Double)] =
      lrClassifier.classify(w, test)

    override def getObjective(): Double =
      lrClassifier.getObjective()

    override def getObjective(w: DenseVector[Double], x: RDD[LabeledPoint]): Double =
      lrClassifier.getObjective(w, x)
  }

}
