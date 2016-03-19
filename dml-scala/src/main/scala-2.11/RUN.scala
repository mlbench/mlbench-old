import org.apache.spark.mllib.classification._
import org.apache.spark.mllib.optimization.{L1Updater, SimpleUpdater, SquaredL2Updater, Updater}
import Functions._
import breeze.linalg.DenseVector
import org.apache.log4j.{Level, Logger}

//Load function
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD

import org.apache.spark._

object RUN {

  def main(args: Array[String]) {
    val conf = new SparkConf().setAppName("Simple Application").setMaster("local[*]")
    val sc = new SparkContext(conf)
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    val numSlices = if (args.length > 0) args(0).toInt else 2

    //Turn off logs
    val rootLogger = Logger.getRootLogger()
    rootLogger.setLevel(Level.ERROR)

    //Load data
    val data : RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc,
      "/Users/amirreza/workspace/distributed-ML-benchmark/dml-scala/dataset/iris.scale.txt")

    //Take only two class with labels -1 and +1 for binary classification
    val points = data.filter(p => p.label == 3.0 || p.label == 2.0).
      map(p => if (p.label == 2.0) LabeledPoint( -1.0, p.features)
               else LabeledPoint( +1.0, p.features)).cache()

    val it = 100
    val lambda = 0.1
    val reg = new L2Regularizer

    runWithMllib(points, reg, lambda, it)

    val eval1 = new Evaluation(new BinaryLogistic,regularizer = reg, lambda = lambda)
    val LR = new LogisticRegression(points, regularizer = reg, lambda = lambda)
    val w = LR.train()
    val objective1 = eval1.getObjective(w, points)
    println("Logistic Objective value: " + objective1)

    val eval2 = new Evaluation(new HingeLoss,regularizer = reg, lambda = lambda)
    val svm = new SVM(points, regularizer = reg, lambda = lambda)
    val w2 = svm.train()
    val object2 = eval2.getObjective(w2, points)
    println("SVM Ovjective value: "+ object2)


    sc.stop()
  }

  def runWithMllib(data : RDD[LabeledPoint],
                   regularizer: Regularizer,
                   lambda: Double,
                   iterations: Int): Unit ={

    val reg: Updater = (regularizer:AnyRef) match {
      case _: L2Regularizer => new SquaredL2Updater
      case _: Unregularized => new SimpleUpdater
    }
    val training = data.map(p => if (p.label == - 1.0) LabeledPoint(0.0, p.features)
      else LabeledPoint(1.0, p.features)).cache()


    val lr = new LogisticRegressionWithLBFGS()
    lr.setIntercept(false)
    lr.optimizer.
      setNumIterations(iterations).
      setRegParam(lambda).
      setUpdater(reg)
    val lrModel = lr.run(training)

    val eval2 = new Evaluation(new BinaryLogistic,regularizer = regularizer, lambda = lambda)
    val object2 = eval2.getObjective(DenseVector(lrModel.weights.toArray), training)
    println("Mllib Logistic w: " + DenseVector(lrModel.weights.toArray))
    println("MLlib Logistic Ovjective value: "+ object2)


    val svm = new SVMWithSGD()
    svm.setIntercept(false)
    svm.optimizer.
      setNumIterations(iterations).
      setRegParam(lambda).
      setUpdater(reg)
    val svmModel = svm.run(training)


    val eval = new Evaluation(new HingeLoss, regularizer, lambda = lambda)
    val object1 = eval.getObjective(DenseVector(svmModel.weights.toArray), training)
    println("Mllib SVM w: " + DenseVector(svmModel.weights.toArray))
    println("MLlib SVM Ovjective value: "+ object1)



  }
}