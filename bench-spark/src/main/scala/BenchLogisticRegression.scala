import Classification.L2_LR_SGD
import org.apache.spark.mllib.optimization.{L1Updater, SimpleUpdater, SquaredL2Updater, Updater}
import Functions._
import org.apache.log4j.{Level, Logger}

//Load function
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.util.SizeEstimator
import org.apache.spark._


object BenchLogisticRegression {

  def main(args: Array[String]) {
    var file = ""
    var parts = 2

    if (args.length > 0) {
      file = args(0)
      parts = args(1).toInt
    }

    //Spark conf
    val conf = new SparkConf().setAppName("Distributed Machine Learning").setMaster("local[*]")
    val sc = new SparkContext(conf)
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    //val numSlices = if (args.length > 0) args(0).toInt else 2

    //Turn off logs
    val rootLogger = Logger.getRootLogger()
    rootLogger.setLevel(Level.ERROR)

    //Load data
    val rawData: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, file)
    val data = rawData.repartition(parts)
    

    //Set optimizer's parameters
    val params = new SGDParameters(
      stepSize = 0.1,
      iterations = 2
    )

    val lambda = 0.1
    val lr = new L2_LR_SGD(lambda, params)

    val start = System.nanoTime()
    //val w1 = lr.train(data)
    val elap = System.nanoTime() - start
    //val objective1 = lr.getObjective(w1.toDenseVector, data)

    //println("Logistic w: " + w1)
    //println("Logistic Objective value: " + objective1)
    println("Training took: " + elap / 1000 / 1000 + "ms")
    println("Number of partitions: " + data.partitions.size)
    println("Size of data rdd: " + SizeEstimator.estimate(data) + " bytes.")
    println("----------------------------")
  }
}
