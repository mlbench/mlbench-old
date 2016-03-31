import java.io.Serializable

import Functions.{Classifier, LossFunction, _}
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
  }

  class LinearClassifier(loss: LossFunction with Classifier,
                         regularizer: Regularizer,
                         params: Parameters)
    extends LinearMethod(loss, regularizer, params) with Classification {

    override def train(data: RDD[LabeledPoint]): DenseVector[Double] = {
      this.optimize(data)
    }

    override def classify(w: DenseVector[Double], test: RDD[LabeledPoint]): RDD[(Double, Double)] = {
      val predictions: RDD[(Double, Double)] = test.map(p => (loss.classify(w.dot(DenseVector(p.features.toArray))), p.label))
      return predictions
    }

    override def predict(w: DenseVector[Double], test: RDD[LabeledPoint]): RDD[(Double, Double)] = {
      return this.classify(w, test)
    }

    override def error(predictions: RDD[(Double, Double)]): Double = {
      predictions.map(p => if (p._1 != p._2) 1.0 else 0.0).reduce(_ + _) / predictions.count()
    }

  }

  class SVM(regularizer: Regularizer = new Unregularized, //No regularizer term by default:
            params: Parameters = new Parameters)
    extends LinearClassifier(new HingeLoss, regularizer, params) with Serializable {
  }

  class LogisticRegression(regularizer: Regularizer = new Unregularized, //No regularizer term by default:
                           params: Parameters = new Parameters)
    extends LinearClassifier(new BinaryLogistic, regularizer, params) with Serializable {

  }

}
