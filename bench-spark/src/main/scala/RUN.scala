import Classification._
import Regression._
import org.apache.log4j.{Level, Logger}
import utils.Utils
import org.rogach.scallop._
import org.apache.spark._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import java.io._

import utils.Functions.ElasticNet

class RunParser(arguments: Seq[String]) extends org.rogach.scallop.ScallopConf(arguments) {
  val dir = opt[String](required = true, default = Some("../results/"), short = 'w', descr = "working directory where results " +
    "are stored. Default is \"../results\". ")
  val optimizers = trailArg[List[String]](descr = "List of optimizers to be used. At least one is required")
  verify()
}

object RUN {

  def main(args: Array[String]) {
    //Spark conf
    val a: Map[String, Object] = null
    val conf = new SparkConf().setAppName("Distributed Machine Learning").setMaster("local[*]")
    val sc = new SparkContext(conf)
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    //Turn off logs
    val rootLogger = Logger.getRootLogger()
    rootLogger.setLevel(Level.ERROR)
    //Parse arguments
    val parser = new RunParser(args)
    val optimizers: List[String] = parser.optimizers()
    assert(optimizers.length > 0)
    val workingDir = parser.dir()

    val train: RDD[LabeledPoint] = Utils.loadLibSVMFromDir(workingDir + "train/", sc)
    val test: RDD[LabeledPoint] = Utils.loadLibSVMFromDir(workingDir + "test/", sc)

    val gdparams = Utils.readGDParameters(workingDir)
    val sgdparams = Utils.readSGDParameters(workingDir)
    val lbfgsparams = Utils.readLBFGSParameters(workingDir)
    val cocoaparams = Utils.readCocoaParameters(workingDir, train, test)
    val elasticparams = Utils.readElasticProxCocoaParameters(workingDir, train, test)
    val l1cocoaparams = Utils.readL1ProxCocoaParameters(workingDir, train, test)
    val lambda = Utils.readRegParameters(workingDir)

    val output = new File(workingDir + "res.out")
    val bw = new BufferedWriter(new FileWriter(output))
    //Run all optimisers given in the args
    optimizers.foreach { opt => opt match {
      case "Elastic_ProxCocoa" => {
        val l1net = new Elastic_ProxCOCOA(train, test, elasticparams, Utils.defaultDebugProxCocoa(train, test))
        val w = l1net.fit()
        bw.write("Elastic_ProxCocoa: " + "lambda: " +
          l1net.regularizer.lambda + " alpha: " + l1net.regularizer.asInstanceOf[ElasticNet].alpha + " elapsed: " +
          l1net.elapsed.get / 1000 / 1000 + "ms " + w.toDenseVector)
        bw.newLine()
        println("Elastic_ProxCocoa Training took: " + l1net.elapsed.get / 1000 / 1000 + "ms")
        println("----------------------------")
      }
      case "L1_Lasso_ProxCocoa" => {
        val l1lasso = new L1_Lasso_ProxCocoa(train, test, l1cocoaparams, Utils.defaultDebugProxCocoa(train, test))
        val w = l1lasso.fit()
        bw.write("L1_Lasso_ProxCocoa: " + "lambda: " +
          l1lasso.regularizer.lambda + " elapsed: " + l1lasso.elapsed.get / 1000 / 1000 + "ms " + w.toDenseVector )
        bw.newLine()
        println("L1_Lasso_ProxCocoa Training took: " + l1lasso.elapsed.get / 1000 / 1000 + "ms")
        println("----------------------------")
      }
      case "L1_Lasso_GD" => {
        val l1lasso = new L1_Lasso_GD(train, params = gdparams, lambda = lambda)
        val w = l1lasso.fit()
        bw.write("L1_Lasso_GD: " + "lambda: " +
          l1lasso.regularizer.lambda + " elapsed: " + l1lasso.elapsed.get / 1000 / 1000 + "ms " + w.toDenseVector )
        bw.newLine()
        println("L1_Lasso_GD Training took: " + l1lasso.elapsed.get / 1000 / 1000 + "ms")
        println("----------------------------")
      }
      case "L1_Lasso_SGD" => {
        val l1lasso = new L1_Lasso_SGD(train, params = sgdparams, lambda = lambda)
        val w = l1lasso.fit()
        bw.write("L1_Lasso_SGD: " + "lambda: " +
          l1lasso.regularizer.lambda + " elapsed: " + l1lasso.elapsed.get / 1000 / 1000 + "ms " + w.toDenseVector )
        bw.newLine()
        println("L1_Lasso_SGD Training took: " + l1lasso.elapsed.get / 1000 / 1000 + "ms")
        println("----------------------------")
      }
      case "L2_SVM_SGD" => {
        val l2svm = new L2_SVM_SGD(train, params = sgdparams, lambda = lambda)
        val w = l2svm.train()
        bw.write("L2_SVM_SGD: " + "lambda: " +
          l2svm.regularizer.lambda + " elapsed: " + l2svm.elapsed.get / 1000 / 1000 + "ms " + w.toDenseVector )
        bw.newLine()
        println("L2_SVM_SGD Training took: " + l2svm.elapsed.get / 1000 / 1000 + "ms")
        println("----------------------------")
      }
      case "L2_SVM_GD" => {
        val l2svm = new L2_SVM_GD(train, params = gdparams, lambda = lambda)
        val w = l2svm.train()
        bw.write("L2_SVM_GD: " + "lambda: " +
          l2svm.regularizer.lambda + " elapsed: " + l2svm.elapsed.get / 1000 / 1000 + "ms " + w.toDenseVector )
        bw.newLine()
        println("L2_SVM_GD Training took: " + l2svm.elapsed.get / 1000 / 1000 + "ms")
        println("----------------------------")
      }
      case "L2_LR_SGD" => {
        val l2lr = new L2_LR_SGD(train, params = sgdparams, lambda = lambda)
        val w = l2lr.train()
        bw.write("L2_LR_SGD: " + "lambda: " +
          l2lr.regularizer.lambda + " elapsed: " + l2lr.elapsed.get / 1000 / 1000 + "ms " + w.toDenseVector )
        bw.newLine()
        println("L2_LR_SGD Training took: " + l2lr.elapsed.get / 1000 / 1000 + "ms")
        println("----------------------------")
      }
      case "L2_LR_GD" => {
        val l2lr = new L2_LR_GD(train, params = gdparams, lambda = lambda)
        val w = l2lr.train()
        bw.write("L2_LR_GD: " + "lambda: " +
          l2lr.regularizer.lambda + " elapsed: " + l2lr.elapsed.get / 1000 / 1000 + "ms " + w.toDenseVector )
        bw.newLine()
        println("L2_LR_GD Training took: " + l2lr.elapsed.get / 1000 / 1000 + "ms")
        println("----------------------------")
      }
      case "L1_LR_SGD" => {
        val l1lr = new L1_LR_SGD(train, params = sgdparams, lambda = lambda)
        val w = l1lr.train()
        bw.write("L1_LR_SGD: " + "lambda: " +
          l1lr.regularizer.lambda + " elapsed: " + l1lr.elapsed.get / 1000 / 1000 + "ms " + w.toDenseVector )
        bw.newLine()
        println("L1_LR_SGD Training took: " + l1lr.elapsed.get / 1000 / 1000 + "ms")
        println("----------------------------")
      }
      case "L1_LR_GD" => {
        val l1lr = new L1_LR_GD(train, params = gdparams, lambda = lambda)
        val w = l1lr.train()
        bw.write("L1_LR_GD: " + "lambda: " +
          l1lr.regularizer.lambda + " elapsed: " + l1lr.elapsed.get / 1000 / 1000 + "ms " + w.toDenseVector )
        bw.newLine()
        println("L1_LR_GD Training took: " + l1lr.elapsed.get / 1000 / 1000 + "ms")
        println("----------------------------")
      }
      case "L2_SVM_Cocoa" => {
        val l2svm = new L2_SVM_COCOA(train, test, cocoaparams, Utils.defaultDebugCocoa(train, test),false)
        val w = l2svm.train()
        bw.write("L2_SVM_Cocoa: " + "lambda: " +
          l2svm.regularizer.lambda + " elapsed: " + l2svm.elapsed.get / 1000 / 1000 + "ms " + w.toDenseVector )
        bw.newLine()
        println("L2_SVM_Cocoa Training took: " + l2svm.elapsed.get / 1000 / 1000 + "ms")
        println("----------------------------")
      }
      case "Mllib_Lasso_SGD" => {
        val lasso = new Mllib_Lasso_SGD(train, params = sgdparams, lambda = lambda)
        val w = lasso.fit()
        bw.write("Mllib_Lasso_SGD: " + "lambda: " +
          lasso.regularizer.lambda + " elapsed: " + lasso.elapsed.get / 1000 / 1000 + "ms " + w.toDenseVector )
        bw.newLine()
        println("Mllib_Lasso_SGD Training took: " + lasso.elapsed.get / 1000 / 1000 + "ms")
        println("----------------------------")
      }
      case "Mllib_Lasso_GD" => {
        val lasso = new Mllib_Lasso_GD(train, params = gdparams, lambda = lambda)
        val w = lasso.fit()
        bw.write("Mllib_Lasso_GD: " + "lambda: " +
          lasso.regularizer.lambda + " elapsed: " + lasso.elapsed.get / 1000 / 1000 + "ms " + w.toDenseVector )
        bw.newLine()
        println("Mllib_Lasso_GD Training took: " + lasso.elapsed.get / 1000 / 1000 + "ms")
        println("----------------------------")
      }
      case "Mllib_L2_LR_LBFGS" => {
        val lbfgs = new Mllib_L2_LR_LBFGS(train, params = lbfgsparams, lambda = lambda)
        val w = lbfgs.train()
        bw.write("Mllib_L2_LR_LBFGS: " + "lambda: " +
          lbfgs.regularizer.lambda + " elapsed: " + lbfgs.elapsed.get / 1000 / 1000 + "ms " + w.toDenseVector )
        bw.newLine()
        println("Mllib_L2_LR_LBFGS Training took: " + lbfgs.elapsed.get / 1000 / 1000 + "ms")
        println("----------------------------")
      }
      case "Mllib_L1_LR_LBFGS" => {
        val lbfgs = new Mllib_L1_LR_LBFGS(train, params = lbfgsparams, lambda = lambda)
        val w = lbfgs.train()
        bw.write("Mllib_L1_LR_LBFGS: " + "lambda: " +
          lbfgs.regularizer.lambda + " elapsed: " + lbfgs.elapsed.get / 1000 / 1000 + "ms " + w.toDenseVector )
        bw.newLine()
        println("Mllib_L1_LR_LBFGS Training took: " + lbfgs.elapsed.get / 1000 / 1000 + "ms")
        println("----------------------------")
      }

      case "Mllib_L1_LR_GD" => {
        val l1lr = new Mllib_L1_LR_GD(train, params = gdparams, lambda = lambda)
        val w = l1lr.train()
        bw.write("Mllib_L1_LR_GD: " + "lambda: " +
          l1lr.regularizer.lambda + " elapsed: " + l1lr.elapsed.get / 1000 / 1000 + "ms " + w.toDenseVector )
        bw.newLine()
        println("Mllib_L1_LR_GD Training took: " + l1lr.elapsed.get / 1000 / 1000 + "ms")
        println("----------------------------")
      }
      case "Mllib_L1_LR_SGD" => {
        val l1lr = new Mllib_L1_LR_SGD(train, params = sgdparams, lambda = lambda)
        val w = l1lr.train()
        bw.write("Mllib_L1_LR_SGD: " + "lambda: " +
          l1lr.regularizer.lambda + " elapsed: " + l1lr.elapsed.get / 1000 / 1000 + "ms " + w.toDenseVector )
        bw.newLine()
        println("Mllib_L1_LR_SGD Training took: " + l1lr.elapsed.get / 1000 / 1000 + "ms")
        println("----------------------------")
      }
      case "Mllib_L2_LR_SGD" => {
        val l1lr = new Mllib_L2_LR_SGD(train, params = sgdparams, lambda = lambda)
        val w = l1lr.train()
        bw.write("Mllib_L2_LR_SGD: " + "lambda: " +
          l1lr.regularizer.lambda + " elapsed: " + l1lr.elapsed.get / 1000 / 1000 + "ms " + w.toDenseVector )
        bw.newLine()
        println("Mllib_L2_LR_SGD Training took: " + l1lr.elapsed.get / 1000 / 1000 + "ms")
        println("----------------------------")
      }
      case "Mllib_L2_LR_GD" => {
        val l1lr = new Mllib_L2_LR_GD(train, params = gdparams, lambda = lambda)
        val w = l1lr.train()
        bw.write("Mllib_L2_LR_GD: " + "lambda: " +
          l1lr.regularizer.lambda + " elapsed: " + l1lr.elapsed.get / 1000 / 1000 + "ms " + w.toDenseVector )
        bw.newLine()
        println("Mllib_L2_LR_GD Training took: " + l1lr.elapsed.get / 1000 / 1000 + "ms")
        println("----------------------------")
      }
      case "Mllib_L2_SVM_SGD" => {
        val l1lr = new Mllib_L2_SVM_SGD(train, params = sgdparams, lambda = lambda)
        val w = l1lr.train()
        bw.write("Mllib_L2_SVM_SGD: " + "lambda: " +
          l1lr.regularizer.lambda + " elapsed: " + l1lr.elapsed.get / 1000 / 1000 + "ms " + w.toDenseVector )
        bw.newLine()
        println("Mllib_L2_SVM_SGD Training took: " + l1lr.elapsed.get / 1000 / 1000 + "ms")
        println("----------------------------")
      }

      case "Mllib_L2_SVM_GD" => {
        val l1lr = new Mllib_L2_SVM_GD(train, params = gdparams, lambda = lambda)
        val w = l1lr.train()
        bw.write("Mllib_L2_SVM_GD: " + "lambda: " +
          l1lr.regularizer.lambda + " elapsed: " + l1lr.elapsed.get / 1000 / 1000 + "ms " + w.toDenseVector )
        bw.newLine()
        println("Mllib_L2_SVM_GD Training took: " + l1lr.elapsed.get / 1000 / 1000 + "ms")
        println("----------------------------")
      }
      case _ => println("The optimizer " + opt + " doest not exist")
    }
    }
    bw.close()
    val outlog = new File(workingDir + "log.out")
    val log = new BufferedWriter(new FileWriter(outlog))
    log.write("GD: " + gdparams.toString)
    log.newLine()
    log.write("SGD: " + sgdparams.toString)
    log.newLine()
    log.write("LBFGS:" + lbfgsparams.toString)
    log.newLine()
    log.write("Cocoa:" + cocoaparams.toString)
    log.newLine()
    log.write("ElasticProx:" + elasticparams.toString)
    log.newLine()
    log.write("L1LassoProx" + l1cocoaparams.toString)
    log.close()
    sc.stop()
  }
}
