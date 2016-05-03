import java.io.Serializable

import optimizers.{Cocoa, SGD, SGDParameters}
import org.apache.spark.rdd.RDD
import breeze.linalg.{DenseVector, Vector}
import distopt.utils.{DebugParams, Params}
import org.apache.spark.mllib.regression.LabeledPoint
import utils.Functions._
import utils.Utils

/**
  * Created by amirreza on 31/03/16.
  */
object Classification {


  trait Classification extends Serializable {
    def train(): Vector[Double]

    def classify(w: Vector[Double], test: RDD[org.apache.spark.mllib.linalg.Vector]): RDD[Double]
  }

  abstract class LinearClassifier(loss: LossFunction with Classifier,
                                  regularizer: Regularizer)
    extends LinearMethod(loss, regularizer) with Classification {

    override def train(): Vector[Double] = {
      this.optimize()
    }

    override def classify(w: Vector[Double], test: RDD[org.apache.spark.mllib.linalg.Vector]): RDD[Double] = {
      val predictions: RDD[Double] = test.map(p => loss.classify(w.dot(DenseVector(p.toArray))))
      return predictions
    }

    override def predict(w: Vector[Double], test: RDD[org.apache.spark.mllib.linalg.Vector]): RDD[Double] = {
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

  class L2_SVM_SGD(data: RDD[LabeledPoint],
                   lambda: Double = 0.01,
                   params: SGDParameters = new SGDParameters(miniBatchFraction = 0.5))
    extends LinearClassifier(new HingeLoss, new L2Regularizer(lambda)) with Serializable {
    val optimizer = new SGD(data, loss, regularizer, params)
    require(params.miniBatchFraction < 1.0, "miniBatchFraction must be less than 1. Use GD otherwise.")
  }

  class L2_SVM_GD(data: RDD[LabeledPoint],
                  lambda: Double = 0.01,
                  params: SGDParameters = new SGDParameters(miniBatchFraction = 1.0))
    extends LinearClassifier(new HingeLoss, new L2Regularizer(lambda)) with Serializable {
    val optimizer = new SGD(data, loss, regularizer, params)
    require(params.miniBatchFraction == 1.0, "Use optimizers.SGD for miniBatchFraction less than 1.0")
  }

  class L2_LR_SGD(data: RDD[LabeledPoint],
                  lambda: Double = 0.01,
                  params: SGDParameters = new SGDParameters(miniBatchFraction = 0.5))
    extends LinearClassifier(new BinaryLogistic, new L2Regularizer(lambda)) with Serializable {
    val optimizer = new SGD(data, loss, regularizer, params)
    require(params.miniBatchFraction < 1.0, "miniBatchFraction must be less than 1. Use GD otherwise.")
  }

  class L2_LR_GD(data: RDD[LabeledPoint],
                 lambda: Double = 0.01,
                 params: SGDParameters = new SGDParameters(miniBatchFraction = 1.0))
    extends LinearClassifier(new BinaryLogistic, new L2Regularizer(lambda)) with Serializable {
    val optimizer = new SGD(data, loss, regularizer, params)
    require(params.miniBatchFraction == 1.0, "Use optimizers.SGD for miniBatchFraction less than 1.0")
  }


  class L1_LR_SGD(data: RDD[LabeledPoint],
                  lambda: Double = 0.01,
                  params: SGDParameters = new SGDParameters(miniBatchFraction = 0.5))
    extends LinearClassifier(new BinaryLogistic, new L1Regularizer(lambda)) with Serializable {
    val optimizer = new SGD(data, loss, regularizer, params)
    require(params.miniBatchFraction < 1.0, "miniBatchFraction must be less than 1. Use GD otherwise.")
  }

  class L1_LR_GD(data: RDD[LabeledPoint],
                 lambda: Double = 0.01,
                 params: SGDParameters = new SGDParameters(miniBatchFraction = 1.0))
    extends LinearClassifier(new BinaryLogistic, new L1Regularizer(lambda)) with Serializable {
    val optimizer = new SGD(data, loss, regularizer, params)
    require(params.miniBatchFraction == 1.0, "Use optimizers.SGD for miniBatchFraction less than 1.0")
  }

  class L2_SVM_COCOA(train: RDD[LabeledPoint],
                     test: RDD[LabeledPoint],
                     params: Params,
                     debug: DebugParams,
                     plus: Boolean)
    extends LinearClassifier(new HingeLoss, new L2Regularizer(params.lambda)) with Serializable {

    def this(train: RDD[LabeledPoint],
             test: RDD[LabeledPoint],
             plus:Boolean){
      this(train, test, Utils.defaultCocoa(train, test)._1, Utils.defaultCocoa(train, test)._2, plus)
    }
    val cocoaData = Utils.toCocoaFormat(train)
    val optimizer = new Cocoa(cocoaData, loss, regularizer, params, debug, plus)

  }

}
