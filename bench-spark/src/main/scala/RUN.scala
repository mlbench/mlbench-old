import Classification._
import Regression.{Elastic_ProxCOCOA, L1_Lasso_GD, L1_Lasso_ProxCocoa, L1_Lasso_SGD}
import breeze.linalg.{DenseVector, SparseVector}
import l1distopt.utils.{DebugParams, Params}
import optimizers.SGDParameters
import org.apache.log4j.{Level, Logger}
import utils.{Evaluation, Utils}
import org.rogach.scallop._
import org.apache.spark._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import java.io._

class Parser(arguments: Seq[String]) extends org.rogach.scallop.ScallopConf(arguments) {
  val dataset = opt[String](required = true, short = 'd',
    descr = "absolute address of the libsvm dataset. This must be provided.")
  val partitions = opt[Int](required = false, default = Some(4), short = 'p', validate = (0 <),
    descr = "Number of spark partitions to be used. Optional.")
  val out = opt[String](default = Some("results"), short = 'o', descr = "The name of the ouput file. Optional.")
  val optimizers = trailArg[List[String]](descr = "List of optimizers to be used. At least one is required")
  val method = opt[String](required = true, short = 'm',
    descr = "Method can be either \"Regression\" or \"Classification\". This must be provided")
  verify()
}

object RUN {

  def main(args: Array[String]) {
    //Spark conf
    val conf = new SparkConf().setAppName("Distributed Machine Learning").setMaster("local[*]")
    val sc = new SparkContext(conf)
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    //Turn off logs
    val rootLogger = Logger.getRootLogger()
    rootLogger.setLevel(Level.ERROR)
    //Parse arguments
    val parser = new Parser(args)
    val optimizers: List[String] = parser.optimizers()
    assert(optimizers.length > 0)
    val dataset = parser.dataset()
    val outname = parser.out()
    val numPartitions = parser.partitions()
    val method = parser.method()
    //Load data
    val (train, test) = method match {
      case "Classification" => Utils.loadAbsolutLibSVMForBinaryClassification(dataset, numPartitions, sc)
      case "Regression" => Utils.loadAbsolutLibSVMForRegression(dataset, numPartitions, sc)
      case _ => throw new IllegalArgumentException("The method " + method + " is not supported.")
    }
    val output = new File(outname + ".out")
    val bw = new BufferedWriter(new FileWriter(output))
    //Run all optimisers given in the args
    optimizers.foreach { opt => opt match {
      case "Elastic_ProxCocoa" => {
        if (method == "Classification") throw new IllegalArgumentException("ElasticNet is a Regression method.")
        val l1net = new Elastic_ProxCOCOA(train, test)
        val w = l1net.fit()
        bw.write("Elastic_ProxCocoa: " + w + " elapsed: " + l1net.elapsed.get / 1000 / 1000 + "ms")
        bw.newLine()
        val objective = l1net.getObjective(w.toDenseVector, train)
        val error1 = l1net.testError(w, test.map(p => p.features), test.map(p => p.label))
        println("Training took: " + l1net.elapsed.get / 1000 / 1000 + "ms")
        println("elastic w: " + w)
        println("elastic Objective value: " + objective)
        println("elastic test error: " + error1)
        println("----------------------------")
      }
      case "L1_Lasso_ProxCocoa" => {
        if (method == "Classification") throw new IllegalArgumentException("L1_Lasso_ProxCocoa is a Regression method.")
        val l1lasso = new L1_Lasso_ProxCocoa(train, test)
        val w = l1lasso.fit()
        bw.write("L1_Lasso_ProxCocoa: " + w + " elapsed: " + l1lasso.elapsed.get / 1000 / 1000 + "ms")
        bw.newLine()
        val objective = l1lasso.getObjective(w.toDenseVector, train)
        val error1 = l1lasso.testError(w, test.map(p => p.features), test.map(p => p.label))
        println("Training took: " + l1lasso.elapsed.get / 1000 / 1000 + "ms")
        println("prox w: " + w)
        println("prox Objective value: " + objective)
        println("prox test error: " + error1)
        println("----------------------------")
      }
      case "L1_Lasso_GD" => {
        if (method == "Classification") throw new IllegalArgumentException("L1_Lasso is a Regression method.")
        val l1lasso = new L1_Lasso_GD(train)
        val w = l1lasso.fit()
        bw.write("L1_Lasso_GD: " + w + " elapsed: " + l1lasso.elapsed.get / 1000 / 1000 + "ms")
        bw.newLine()
        val objective = l1lasso.getObjective(w.toDenseVector, train)
        val error1 = l1lasso.testError(w, test.map(p => p.features), test.map(p => p.label))
        println("Training took: " + l1lasso.elapsed.get / 1000 / 1000 + "ms")
        println("L1_Lasso_GD w: " + w)
        println("L1_Lasso_GD Objective value: " + objective)
        println("L1_Lasso_GD test error: " + error1)
        println("----------------------------")
      }
      case "L1_Lasso_SGD" => {
        if (method == "Classification") throw new IllegalArgumentException("L1_Lasso is a Regression method.")
        val l1lasso = new L1_Lasso_SGD(train)
        val w = l1lasso.fit()
        bw.write("L1_Lasso_SGD: " + w + " elapsed: " + l1lasso.elapsed.get / 1000 / 1000 + "ms")
        bw.newLine()
        val objective = l1lasso.getObjective(w.toDenseVector, train)
        val error = l1lasso.testError(w, test.map(p => p.features), test.map(p => p.label))
        println("Training took: " + l1lasso.elapsed.get / 1000 / 1000 + "ms")
        println("L1_Lasso_SGD w: " + w)
        println("L1_Lasso_SGD Objective value: " + objective)
        println("L1_Lasso_SGD test error: " + error)
        println("----------------------------")
      }
      case "L2_SVM_SGD" => {
        if (method == "Regression") throw new IllegalArgumentException("L2_SVM is a Classification method.")
        val l2svm = new L2_SVM_SGD(train)
        val w = l2svm.train()
        bw.write("L2_SVM_SGD: " + w + " elapsed: " + l2svm.elapsed.get / 1000 / 1000 + "ms")
        bw.newLine()
        val objective = l2svm.getObjective(w.toDenseVector, train)
        val error = l2svm.testError(w, test.map(p => p.features), test.map(p => p.label))
        println("Training took: " + l2svm.elapsed.get / 1000 / 1000 + "ms")
        println("L2_SVM_SGD w: " + w)
        println("L2_SVM_SGD Objective value: " + objective)
        println("L2_SVM_SGD test error: " + error)
        println("----------------------------")
      }
      case "L2_SVM_GD" => {
        if (method == "Regression") throw new IllegalArgumentException("L2_SVM is a Classification method.")
        val l2svm = new L2_SVM_GD(train)
        val w = l2svm.train()
        bw.write("L2_SVM_GD: " + w + " elapsed: " + l2svm.elapsed.get / 1000 / 1000 + "ms")
        bw.newLine()
        val objective = l2svm.getObjective(w.toDenseVector, train)
        val error = l2svm.testError(w, test.map(p => p.features), test.map(p => p.label))
        println("Training took: " + l2svm.elapsed.get / 1000 / 1000 + "ms")
        println("L2_SVM_GD w: " + w)
        println("L2_SVM_GD Objective value: " + objective)
        println("L2_SVM_GD test error: " + error)
        println("----------------------------")
      }
      case "L2_LR_SGD" => {
        if (method == "Regression") throw new IllegalArgumentException("L2_LR is a Classification method.")
        val l2lr = new L2_LR_SGD(train)
        val w = l2lr.train()
        bw.write("L2_LR_SGD: " + w + " elapsed: " + l2lr.elapsed.get / 1000 / 1000 + "ms")
        bw.newLine()
        val objective = l2lr.getObjective(w.toDenseVector, train)
        val error = l2lr.testError(w, test.map(p => p.features), test.map(p => p.label))
        println("Training took: " + l2lr.elapsed.get / 1000 / 1000 + "ms")
        println("L2_LR_SGD w: " + w)
        println("L2_LR_SGD Objective value: " + objective)
        println("L2_LR_SGD test error: " + error)
        println("----------------------------")
      }
      case "L2_LR_GD" => {
        if (method == "Regression") throw new IllegalArgumentException("L2_LR is a Classification method.")
        val l2lr = new L2_LR_GD(train)
        val w = l2lr.train()
        bw.write("L2_LR_GD: " + w + " elapsed: " + l2lr.elapsed.get / 1000 / 1000 + "ms")
        bw.newLine()
        val objective = l2lr.getObjective(w.toDenseVector, train)
        val error = l2lr.testError(w, test.map(p => p.features), test.map(p => p.label))
        println("Training took: " + l2lr.elapsed.get / 1000 / 1000 + "ms")
        println("L2_LR_GD w: " + w)
        println("L2_LR_GD Objective value: " + objective)
        println("L2_LR_GD test error: " + error)
        println("----------------------------")
      }
      case "L1_LR_SGD" => {
        if (method == "Regression") throw new IllegalArgumentException("L1_LR is a Classification method.")
        val l1lr = new L1_LR_SGD(train)
        val w = l1lr.train()
        bw.write("L1_LR_SGD: " + w + " elapsed: " + l1lr.elapsed.get / 1000 / 1000 + "ms")
        bw.newLine()
        val objective = l1lr.getObjective(w.toDenseVector, train)
        val error = l1lr.testError(w, test.map(p => p.features), test.map(p => p.label))
        println("Training took: " + l1lr.elapsed.get / 1000 / 1000 + "ms")
        println("L1_LR_SGD w: " + w)
        println("L1_LR_SGD Objective value: " + objective)
        println("L1_LR_SGD test error: " + error)
        println("----------------------------")
      }
      case "L1_LR_GD" => {
        if (method == "Regression") throw new IllegalArgumentException("L1_LR is a Classification method.")
        val l1lr = new L1_LR_GD(train)
        val w = l1lr.train()
        bw.write("L1_LR_GD: " + w + " elapsed: " + l1lr.elapsed.get / 1000 / 1000 + "ms")
        bw.newLine()
        val objective = l1lr.getObjective(w.toDenseVector, train)
        val error = l1lr.testError(w, test.map(p => p.features), test.map(p => p.label))
        println("Training took: " + l1lr.elapsed.get / 1000 / 1000 + "ms")
        println("L1_LR_GD w: " + w)
        println("L1_LR_GD Objective value: " + objective)
        println("L1_LR_GD test error: " + error)
        println("----------------------------")
      }
      case "L2_SVM_Cocoa" => {
        if (method == "Regression") throw new IllegalArgumentException("L2_SVM is a Classification method.")
        val l2svm = new L2_SVM_COCOA(train, test, false)
        val w = l2svm.train()
        bw.write("L2_SVM_Cocoa: " + w + " elapsed: " + l2svm.elapsed.get / 1000 / 1000 + "ms")
        bw.newLine()
        val objective = l2svm.getObjective(w.toDenseVector, train)
        val error = l2svm.testError(w, test.map(p => p.features), test.map(p => p.label))
        println("Training took: " + l2svm.elapsed.get / 1000 / 1000 + "ms")
        println("L2_SVM_Cocoa w: " + w)
        println("L2_SVM_Cocoa Objective value: " + objective)
        println("L2_SVM_Cocoa test error: " + error)
        println("----------------------------")
      }
      case _ => println("The optimizer " + opt + " doest not exist")
    }
    }
    bw.close()
    sc.stop()
  }
}
