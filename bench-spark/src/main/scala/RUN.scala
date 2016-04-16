import Classification.{L2_LR_SGD, L2_SVM_SGD}
import org.apache.spark.mllib.classification._
import org.apache.spark.mllib.optimization.{L1Updater, SimpleUpdater, SquaredL2Updater, Updater}
import Regression.{Elastic_ProxCOCOA, L1_Lasso_SGD}
import breeze.linalg.{DenseVector, SparseVector}
import l1distopt.utils.{DebugParams, Params}
import optimizers.SGDParameters
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.linalg.Vectors
import utils.Functions._
import utils.{Utils}



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

    //Load classification data
    //val dataset = "iris.scale.txt"
    //val points = Utils.loadLibSVMForBinaryClassification(dataset, numPartitions, sc)
    //val Array(train, test) = points.randomSplit(Array(0.8, 0.2), seed = 13)

    //Load regression data
    val numFeats = 10 //number of features for breast-cancer data
    val filename = "breast-cancer_scale.libsvm"
    val (trainColumn, testColumn, train, test, trainProx, testProx) =
      Utils.loadLibSVMForRegressionProxCocoa( filename , numPartitions, numFeats, sc)

    //Regularization parameters
    val lambda = 0.1
    val eta = 0.5
    //optimization parameters
    val iterations = 100
    val localIterFrac = 0.9
    val debugIter = 10
    val data = trainColumn._1.cache()
    val force_cache = data.count().toInt
    val labels = trainColumn._2
    val n = labels.size
    var localIters = (localIterFrac * numFeats / data.partitions.size).toInt
    localIters = Math.max(localIters,1)
    val alphaInit = SparseVector.zeros[Double](10)
    val proxParams = Params(alphaInit, n, iterations, localIters, lambda, eta)
    val debug = DebugParams(testProx, debugIter, seed)

    val l1net = new Elastic_ProxCOCOA(proxParams, debug)
    val w1 = l1net.fit(trainColumn)
    val objective1 = l1net.getObjective(w1.toDenseVector, train)
    val error1 = l1net.testError(w1, test.map(p => p.features), test.map(p => p.label))
    println("Lasso w: " + w1)
    println("Lasso Objective value: " + objective1)
    println("Lasso test error: " + error1)
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
