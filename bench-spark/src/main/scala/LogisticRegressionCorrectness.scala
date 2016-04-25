import Classification.L2_LR_SGD
import breeze.linalg.DenseVector
import optimizers.SGDParameters
import org.apache.spark.mllib.classification._
import org.apache.spark.mllib.optimization.{L1Updater, SimpleUpdater, SquaredL2Updater, Updater}
import org.apache.log4j.{Level, Logger}
import utils.Evaluation
import utils.Functions.{BinaryLogistic, L2Regularizer, Regularizer, Unregularized}

//Load function
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD

import org.apache.spark._


object LogisticRegressionCorrectness {

  def main(args: Array[String]) {
    //Spark conf
    val conf = new SparkConf().setAppName("Distributed Machine Learning").setMaster("local[*]")
    val sc = new SparkContext(conf)
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)

    //Turn off logs
    val rootLogger = Logger.getRootLogger()
    rootLogger.setLevel(Level.ERROR)

    //Load data
    val file = args(0)
    val data: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, file)
    val Array(train, test) = data.randomSplit(Array(0.8, 0.2), seed = 13)

    //Set optimizer's parameters
    val params = new SGDParameters(
      stepSize = 0.1,
      iterations = 100
    )
    val lambda = 0.1
    val reg = new L2Regularizer(lambda = lambda)

    //Fit with Mllib in order to compare
    runLRWithMllib(train, test, reg, lambda, params.iterations, params.stepSize)
    println("----------------------------")

    //Classify with Binary Logistic Regression
    val lr = new L2_LR_SGD(train, lambda, params)
    val w1 = lr.train()
    val objective1 = lr.getObjective(w1.toDenseVector, data)
    val error1 = lr.testError(w1, test.map(p => p.features), test.map(p => p.label))
    println("Logistic w: " + w1)
    println("Logistic Objective value: " + objective1)
    println("Logistic CV error: " + error1)
    println("----------------------------")

    sc.stop()
  }


  def runLRWithMllib(train: RDD[LabeledPoint],
                     test: RDD[LabeledPoint],
                     regularizer: Regularizer,
                     lambda: Double,
                     iterations: Int,
                     stepSize: Double): Unit = {

    val reg: Updater = (regularizer: AnyRef) match {
      case _: L2Regularizer => new SquaredL2Updater
      case _: Unregularized => new SimpleUpdater
    }
    val training = train.map(p => if (p.label == -1.0) LabeledPoint(0.0, p.features)
    else LabeledPoint(1.0, p.features)).cache()

    //Logistic Regression
    val lr = new LogisticRegressionWithSGD()
    lr.setIntercept(false)
    lr.optimizer.
      setNumIterations(iterations).
      setRegParam(lambda).
      setUpdater(reg).
      setStepSize(stepSize)
    val lrModel = lr.run(training)

    val eval2 = new Evaluation(new BinaryLogistic, regularizer = regularizer, lambda = lambda)
    val object2 = eval2.getObjective(DenseVector(lrModel.weights.toArray), training)
    println("Mllib Logistic w: " + DenseVector(lrModel.weights.toArray))
    println("Mllib Logistic Objective value: " + object2)

    val predictions = test.map { case LabeledPoint(label, features) =>
      lrModel.predict(features)
    }
    val error = eval2.error(test.map(p => p.label), predictions)
    println("Mllib Logistic CV error: " + error)

  }

}
