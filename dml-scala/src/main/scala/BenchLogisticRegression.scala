import Classifications.LogisticRegression
import org.apache.spark.mllib.optimization.{L1Updater, SimpleUpdater, SquaredL2Updater, Updater}
import Functions._
import org.apache.log4j.{Level, Logger}

//Load function
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD

import org.apache.spark._


object BenchLogisticRegression {

  def main(args: Array[String]) {
    var file = ""
    if (args.length > 0)
      file = args(0)

    //Spark conf
    val conf = new SparkConf().setAppName("Distributed Machine Learning").setMaster("local[*]")
    val sc = new SparkContext(conf)
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    //val numSlices = if (args.length > 0) args(0).toInt else 2

    //Turn off logs
    val rootLogger = Logger.getRootLogger()
    rootLogger.setLevel(Level.ERROR)

    //Load data
    val data: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, file)

    //Set optimizer's parameters
    val params = new Parameters(
      stepSize = 0.1,
      iterations = 100,
      lambda = 0.1
    )
    val reg = new L2Regularizer

    val lr = new LogisticRegression(regularizer = reg, params)

    val start = System.nanoTime()
    val w1 = lr.train(data)
    val elap = System.nanoTime() - start
    val objective1 = lr.getObjective(w1, data)

    println("Logistic w: " + w1)
    println("Logistic Objective value: " + objective1)
    println("Training took: " + elap / 1000 / 1000 + "ms")
    println("----------------------------")
  }
}
