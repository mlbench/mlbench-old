import Classification.{L2_LR_SGD, L2_SVM_SGD}
import org.apache.spark.mllib.classification._
import org.apache.spark.mllib.optimization.{L1Updater, SimpleUpdater, SquaredL2Updater, Updater}
import Functions._
import breeze.linalg.DenseVector
import org.apache.log4j.{Level, Logger}
import utils.Utils

//Load function
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

import org.apache.spark._

object RUN {

  def main(args: Array[String]) {
    //Spark conf
    val conf = new SparkConf().setAppName("Distributed Machine Learning").setMaster("local[*]")
    val sc = new SparkContext(conf)
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)

    //Turn off logs
    val rootLogger = Logger.getRootLogger()
    rootLogger.setLevel(Level.ERROR)

    //Load data
    val dataset = "iris.scale.txt"
    val numPartitions = 4
    val points = Utils.loadLibSVMForBinaryClassification(dataset, numPartitions, sc)
    val Array(train, test) = points.randomSplit(Array(0.8, 0.2), seed = 13)
    //Set optimizer's parameters
    val params = new SGDParameters(
      iterations = 100,
      miniBatchFraction = 0.9,
      stepSize = 0.1,
      seed = 13
    )
    //Regularization parameter
    val lambda = 0.1
    val reg = new L2Regularizer(lambda = lambda)

    //Fit with Mllib in order to compare
    runLRWithMllib(train, test, reg, lambda, params.iterations, params.miniBatchFraction, params.stepSize)
    println("----------------------------")
    runSVMWithMllib(train, test, reg, lambda, params.iterations, params.miniBatchFraction, params.stepSize)
    println("----------------------------")

    //Classify with Binary Logistic Regression
    val lr = new L2_LR_SGD(lambda, params)
    val w1 = lr.train(train)
    val objective1 = lr.getObjective()
    val error1 = lr.testError(w1, test.map(p => p.features), test.map(p => p.label))
    println("Logistic w: " + w1)
    println("Logistic Objective value: " + objective1)
    println("Logistic test error: " + error1)
    println("----------------------------")

    //Classify with SVM
    val svm = new L2_SVM_SGD(lambda, params)
    val w2 = svm.train(train)
    val object2 = svm.getObjective()
    val error2 = svm.testError(w2, test.map(p => p.features), test.map(p => p.label))
    println("SVM w: " + w2)
    println("SVM Ovjective value: " + object2)
    println("SVM test error: " + error2)
    println("----------------------------")


    sc.stop()
  }

  def runLRWithMllib(train: RDD[LabeledPoint],
                     test: RDD[LabeledPoint],
                     regularizer: Regularizer,
                     lambda: Double,
                     iterations: Int,
                     fraction: Double,
                     stepSize: Double): Unit = {

    val reg: Updater = (regularizer: Regularizer) match {
      case _: L1Regularizer => new L1Updater
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
      setMiniBatchFraction(fraction).
      setStepSize(stepSize)
    val lrModel = lr.run(training)

    val scores = test.map { point =>
      lrModel.predict(point.features)
    }.map(p => if (p == 0) -1.0 else 1.0)

    val eval = new Evaluation(new BinaryLogistic, regularizer = regularizer, lambda = lambda)
    val objective = eval.getObjective(DenseVector(lrModel.weights.toArray), training)
    val error = eval.error(scores, test.map(p => p.label))
    println("Mllib Logistic w: " + DenseVector(lrModel.weights.toArray))
    println("Mllib Logistic Objective value: " + objective)
    println("Mllib Logistic test error: " + error)

  }

  def runSVMWithMllib(train: RDD[LabeledPoint],
                      test: RDD[LabeledPoint],
                      regularizer: Regularizer,
                      lambda: Double,
                      iterations: Int,
                      fraction: Double,
                      stepSize: Double): Unit = {

    val reg: Updater = (regularizer: AnyRef) match {
      case _: L2Regularizer => new SquaredL2Updater
      case _: Unregularized => new SimpleUpdater
    }
    val training = train.map(p => if (p.label == -1.0) LabeledPoint(0.0, p.features)
    else LabeledPoint(1.0, p.features)).cache()


    //SVM:
    val svm = new SVMWithSGD()
    svm.setIntercept(false)
    svm.optimizer.
      setNumIterations(iterations).
      setMiniBatchFraction(fraction).
      setRegParam(lambda).
      setUpdater(reg).
      setStepSize(stepSize)
    val svmModel = svm.run(training)

    val scores = test.map { point =>
      svmModel.predict(point.features)
    }.map(p => if (p == 0) -1.0 else 1.0)

    val eval = new Evaluation(new HingeLoss, regularizer, lambda = lambda)
    val objective = eval.getObjective(DenseVector(svmModel.weights.toArray), training)
    val error = eval.error(scores, test.map(p => p.label))
    println("Mllib SVM w: " + DenseVector(svmModel.weights.toArray))
    println("Mllib SVM Ovjective value: " + objective)
    println("Mllib SVM test error: " + error)
  }
}
