import Classification.{L2_LR_SGD, L2_SVM_COCOA, L2_SVM_SGD}
import org.apache.spark.mllib.classification._
import org.apache.spark.mllib.optimization.{L1Updater, SimpleUpdater, SquaredL2Updater, Updater}
import Regression.{Elastic_ProxCOCOA, L1_Lasso_SGD}
import breeze.linalg.{DenseVector, SparseVector}
import l1distopt.utils.{DebugParams, Params}
import optimizers.SGDParameters
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.linalg.Vectors
import utils.Functions._
import utils.{Evaluation, Utils}



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

    val numPartitions = 4
    val seed = 13
    //Load regression data
    val filename = "breast-cancer_scale.libsvm"
    val (trainReg , testReg) = Utils.loadLibSVMForRegression( filename , numPartitions, sc)

    //Regularization parameters
    val lambda = 0.1
    val eta = 0.5
    //optimization parameters
    val iterations = 100
    val localIterFrac = 0.9
    val debugIter = 10
    val force_cache = trainReg.count().toInt
    val n = trainReg.count().toInt
    var localIters = (localIterFrac * trainReg.first().features.size / trainReg.partitions.size).toInt
    localIters = Math.max(localIters,1)
    val alphaInit = SparseVector.zeros[Double](10)
    val proxParams = Params(alphaInit, n, iterations, localIters, lambda, eta)
    val debug = DebugParams(Utils.toProxCocoaFormat(testReg), debugIter, seed)

    val l1net = new Elastic_ProxCOCOA(trainReg, proxParams, debug)
    val w1 = l1net.fit()
    val objective1 = l1net.getObjective(w1.toDenseVector, trainReg)
    val error1 = l1net.testError(w1, testReg.map(p => p.features), testReg.map(p => p.label))
    println("prox w: " + w1)
    println("prox Objective value: " + objective1)
    println("prox test error: " + error1)
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
      case _: L1Regularizer => new L1Updater
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
