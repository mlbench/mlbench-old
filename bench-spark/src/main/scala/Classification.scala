import java.io.Serializable

import Functions.{Classifier, LossFunction, _}
import Regression.LinearRegression
import breeze.linalg.DenseVector
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

/**
  * Created by amirreza on 31/03/16.
  */
object Classification {


  trait Classification extends Serializable {
    def train(data: RDD[LabeledPoint]): DenseVector[Double]

    def classify(w: DenseVector[Double], test: RDD[org.apache.spark.mllib.linalg.Vector]): RDD[Double]
  }

  abstract class LinearClassifier(loss: LossFunction with Classifier,
                                  regularizer: Regularizer)
    extends LinearMethod(loss, regularizer) with Classification {

    override def train(data: RDD[LabeledPoint]): DenseVector[Double] = {
      this.optimize(data)
    }

    override def classify(w: DenseVector[Double], test: RDD[org.apache.spark.mllib.linalg.Vector]): RDD[Double] = {
      val predictions: RDD[Double] = test.map(p => loss.classify(w.dot(DenseVector(p.toArray))))
      return predictions
    }

    override def predict(w: DenseVector[Double], test: RDD[org.apache.spark.mllib.linalg.Vector]): RDD[Double] = {
      return this.classify(w, test)
    }
    //Missclassification error
    override def error(true_labels: RDD[Double], predictions: RDD[Double]): Double = {
      predictions.zip(true_labels).map(p => if (p._1 != p._2) 1.0 else 0.0).reduce(_ + _) / predictions.count()
    }

  }

  /*
    Tasks L2:
   */

  class L2_SVM_SGD(lambda: Double = 0.1,
                   params: SGDParameters = new SGDParameters(miniBatchFraction = 0.5))
    extends LinearClassifier(new HingeLoss, new L2Regularizer(lambda)) with Serializable {
    val optimizer: Optimizer = new SGD(loss, regularizer, params)
    require(params.miniBatchFraction < 1.0, "miniBatchFraction must be less than 1. Use GD otherwise.")
  }

  class L2_SVM_GD(lambda: Double = 0.1,
                  params: SGDParameters = new SGDParameters(miniBatchFraction = 1.0))
    extends LinearClassifier(new HingeLoss, new L2Regularizer(lambda)) with Serializable {
    val optimizer: Optimizer = new SGD(loss, regularizer, params)
    require(params.miniBatchFraction == 1.0, "Use SGD for miniBatchFraction less than 1.0")
  }

  class L2_LR_SGD(lambda: Double = 0.1,
                  params: SGDParameters = new SGDParameters(miniBatchFraction = 0.5))
    extends LinearClassifier(new BinaryLogistic, new L2Regularizer(lambda)) with Serializable {
    val optimizer: Optimizer = new SGD(loss, regularizer, params)
    require(params.miniBatchFraction < 1.0, "miniBatchFraction must be less than 1. Use GD otherwise.")
  }

  class L2_LR_GD(lambda: Double = 0.1,
                 params: SGDParameters = new SGDParameters(miniBatchFraction = 0.5))
    extends LinearClassifier(new BinaryLogistic, new L2Regularizer(lambda)) with Serializable {
    val optimizer: Optimizer = new SGD(loss, regularizer, params)
    require(params.miniBatchFraction == 1.0, "Use SGD for miniBatchFraction less than 1.0")
  }


  class L1_LR_SGD(lambda: Double = 0.1,
                  params: SGDParameters = new SGDParameters(miniBatchFraction = 0.5))
    extends LinearClassifier(new BinaryLogistic, new L1Regularizer(lambda)) with Serializable {
    val optimizer: Optimizer = new SGD(loss, regularizer, params)
    require(params.miniBatchFraction < 1.0, "miniBatchFraction must be less than 1. Use GD otherwise.")
  }

  class L1_LR_GD(lambda: Double = 0.1,
                 params: SGDParameters = new SGDParameters(miniBatchFraction = 0.5))
    extends LinearClassifier(new BinaryLogistic, new L1Regularizer(lambda)) with Serializable {
    val optimizer: Optimizer = new SGD(loss, regularizer, params)
    require(params.miniBatchFraction == 1.0, "Use SGD for miniBatchFraction less than 1.0")
  }

}
