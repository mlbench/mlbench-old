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

    //Adjust labels
    val datac = data.map(p => LabeledPoint(p.label - 2, p.features)) //Labels must be non negative integers:0,1,2,...
    val points = datac.filter(p => p.label == 0 || p.label == 1)//Binary classification, take only 0 and 1

    val LR = new LogisticRegression(points)
    val w = LR.train()

    val eval = new Evaluation(R = Functions.none_reg)
    val objective = eval.getObjective(w, points)
    println("Objective value: " + objective)

    val svm = new SVM(points)
    val w2 = svm.train()

    val eval2 = new Evaluation(R = Functions.none_reg)
    val object2 = eval2.getObjective(w2, points)
    println("Ovjective value: "+ object2)

    sc.stop()
  }
}