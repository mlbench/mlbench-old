import java.io.Serializable

import breeze.linalg.DenseVector
import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.apache.spark.{SparkConf, SparkContext}
import utils.Functions._
import utils.Utils

import scala.io.Source

/**
  * Created by amirreza on 09/03/16.
  */

class EvalParser(arguments: Seq[String]) extends org.rogach.scallop.ScallopConf(arguments) {
  val dir = opt[String](required = true, default = Some("../results/"), short = 'd',
    descr = "absolute address of the working directory. This must be provided.")
  verify()
}

object Evaluate {
  def main(args: Array[String]) {
    //Spark conf
    val conf = new SparkConf().setAppName("Distributed Machine Learning").setMaster("local[*]")
    val sc = new SparkContext(conf)
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    //Turn off logs
    val rootLogger = Logger.getRootLogger()
    rootLogger.setLevel(Level.ERROR)

    val parser = new EvalParser(args)
    val workinDir = parser.dir()
    val inputLines = Source.fromFile(workinDir + "res.out").getLines.toList
    val pattern = "^([^:]+): DenseVector\\((.*)\\) elapsed: ([1-9]*)ms lambda: ([0-9]*\\.[0-9]*)".r

    val train: RDD[LabeledPoint] = Utils.loadLibSVMFromDir(workinDir + "train/", sc)

    inputLines.foreach { line =>
      val pattern(method, weights_row, elapsed_row, lambda_raw) = line
      val weights = DenseVector(weights_row.split(",").map(_.trim).map(_.toDouble))
      val elapsed = elapsed_row.toInt
      val lambda = lambda_raw.toDouble
      val alpha = 0.01 //TODO

      method match {
        case "Elastic_ProxCocoa" => {
          val eval = new Evaluation(new SquaredLoss, new ElasticNet(lambda, alpha))
          val objective = eval.getObjective(weights, train)
          println("objective: " + objective)
        }
        case "L1_Lasso_SGD" | "L1_Lasso_GD" | "L1_Lasso_ProxCocoa" => {
          val eval = new Evaluation(new SquaredLoss, new L1Regularizer(lambda))
          val objective = eval.getObjective(weights, train)
          println("objective: " + objective)
        }
        case "L2_LR_SGD" | "L2_LR_GD" => {
          val eval = new Evaluation(new BinaryLogistic, new L2Regularizer(lambda))
          val objective = eval.getObjective(weights, train)
          println("L2_LR objective: " + objective)
        }
        case "L1_LR_SGD" | "L1_LR_SGD" => {
          val eval = new Evaluation(new BinaryLogistic, new L1Regularizer(lambda))
          val objective = eval.getObjective(weights, train)
          println("objective: " + objective)
        }
        case "L2_SVM_Cocoa" | "L2_SVM_GD" | "L2_SVM_SGD" => {
          val eval = new Evaluation(new HingeLoss, new L2Regularizer(lambda))
          val objective = eval.getObjective(weights, train)
          println("objective: " + objective)
        }
        case _ => println("No such method.")
      }
    }
  }
}

class Evaluation(loss: LossFunction = new HingeLoss,
                 regularizer: Regularizer = new Unregularized) extends Serializable {

  def getObjective(w: DenseVector[Double], x: RDD[LabeledPoint]): Double = {
    val n: Double = x.count()
    val sum = x.map(p => loss.loss(w, DenseVector(p.features.toArray), p.label)).reduce(_ + _)
    return regularizer.lambda * regularizer.value(w) + (sum / n);
  }

  def error(true_labels: RDD[Double], predictions: RDD[Double]): Double = {
    predictions.zip(true_labels).map(p => if (p._1 != p._2) 1.0 else 0.0).reduce(_ + _) / predictions.count()
  }
}


