import Classification._
import Regression.{Elastic_ProxCOCOA, L1_Lasso_GD, L1_Lasso_ProxCocoa, L1_Lasso_SGD}
import breeze.linalg.{DenseVector, SparseVector}
import l1distopt.utils.{DebugParams, Params}
import optimizers.SGDParameters
import org.apache.log4j.{Level, Logger}
import utils.{Evaluation, Utils}
import org.rogach.scallop._

import org.apache.spark._

class Parser(arguments: Seq[String]) extends org.rogach.scallop.ScallopConf(arguments) {
  val dataset = opt[String](required = true, short = 'd', descr = "absolute address of the libsvm dataset")
  val partitions = opt[Int](required = false, default = Some(4),
    short = 'p', validate = (0 <), descr = "Number of spark partitions to be used.")
  val out = opt[String](default = Some("out"), short = 'o', descr = "The name of the ouput file")
  val optimizers = trailArg[List[String]](descr = "List of optimizers to be used")
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
    val dataset = parser.dataset()
    val output = parser.out()
    val numPartitions = parser.partitions()
    //Load data
    val (trainReg, testReg) = Utils.loadLibSVMForRegression(dataset, numPartitions, sc)
    val (trainClass, testClass) = Utils.loadLibSVMForBinaryClassification(dataset, numPartitions, sc)

    //Run all optimisers given in the args
    optimizers.foreach { opt => opt match {
      case "Elastic_ProxCocoa" => {
        val seed = 13
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
        localIters = Math.max(localIters, 1)
        val alphaInit = SparseVector.zeros[Double](10)
        val proxParams = Params(alphaInit, n, iterations, localIters, lambda, eta)
        val debug = DebugParams(Utils.toProxCocoaFormat(testReg), debugIter, seed)

        val l1net = new Elastic_ProxCOCOA(trainReg, proxParams, debug)
        val w = l1net.fit()
        val objective = l1net.getObjective(w.toDenseVector, trainReg)
        val error1 = l1net.testError(w, testReg.map(p => p.features), testReg.map(p => p.label))
        println("elastic w: " + w)
        println("elastic Objective value: " + objective)
        println("elastic test error: " + error1)
        println("----------------------------")
      }
      case "L1_Lasso_ProxCocoa" => {
        val seed = 13
        //Regularization parameters
        val lambda = 0.1
        val eta = 1.0
        //optimization parameters
        val iterations = 100
        val localIterFrac = 0.9
        val debugIter = 10
        val force_cache = trainReg.count().toInt
        val n = trainReg.count().toInt
        var localIters = (localIterFrac * trainReg.first().features.size / trainReg.partitions.size).toInt
        localIters = Math.max(localIters, 1)
        val alphaInit = SparseVector.zeros[Double](10)
        val proxParams = Params(alphaInit, n, iterations, localIters, lambda, eta)
        val debug = DebugParams(Utils.toProxCocoaFormat(testReg), debugIter, seed)

        val l1lasso = new L1_Lasso_ProxCocoa(trainReg, proxParams, debug)
        val w = l1lasso.fit()
        val objective = l1lasso.getObjective(w.toDenseVector, trainReg)
        val error1 = l1lasso.testError(w, testReg.map(p => p.features), testReg.map(p => p.label))
        println("prox w: " + w)
        println("prox Objective value: " + objective)
        println("prox test error: " + error1)
        println("----------------------------")
      }
      case "L1_Lasso_GD" => {
        val l1lasso = new L1_Lasso_GD(trainReg)
        val w = l1lasso.fit()
        val objective = l1lasso.getObjective(w.toDenseVector, trainReg)
        val error1 = l1lasso.testError(w, testReg.map(p => p.features), testReg.map(p => p.label))
        println("L1_Lasso_GD w: " + w)
        println("L1_Lasso_GD Objective value: " + objective)
        println("L1_Lasso_GD test error: " + error1)
        println("----------------------------")
      }
      case "L1_Lasso_SGD" => {
        val l1lasso = new L1_Lasso_SGD(trainReg)
        val w = l1lasso.fit()
        val objective = l1lasso.getObjective(w.toDenseVector, trainReg)
        val error = l1lasso.testError(w, testReg.map(p => p.features), testReg.map(p => p.label))
        println("L1_Lasso_SGD w: " + w)
        println("L1_Lasso_SGD Objective value: " + objective)
        println("L1_Lasso_SGD test error: " + error)
        println("----------------------------")
      }
      case "L2_SVM_SGD" => {
        val l2svm = new L2_SVM_SGD(trainClass)
        val w = l2svm.train()
        val objective = l2svm.getObjective(w.toDenseVector, trainClass)
        val error = l2svm.testError(w, testClass.map(p => p.features), testClass.map(p => p.label))
        println("L2_SVM_SGD w: " + w)
        println("L2_SVM_SGD Objective value: " + objective)
        println("L2_SVM_SGD test error: " + error)
        println("----------------------------")
      }
      case "L2_SVM_GD" => {
        val l2svm = new L2_SVM_GD(trainClass)
        val w = l2svm.train()
        val objective = l2svm.getObjective(w.toDenseVector, trainClass)
        val error = l2svm.testError(w, testClass.map(p => p.features), testClass.map(p => p.label))
        println("L2_SVM_GD w: " + w)
        println("L2_SVM_GD Objective value: " + objective)
        println("L2_SVM_GD test error: " + error)
        println("----------------------------")
      }
      case "L2_LR_SGD" => {
        val l2lr = new L2_LR_SGD(trainClass)
        val w = l2lr.train()
        val objective = l2lr.getObjective(w.toDenseVector, trainClass)
        val error = l2lr.testError(w, testClass.map(p => p.features), testClass.map(p => p.label))
        println("L2_LR_SGD w: " + w)
        println("L2_LR_SGD Objective value: " + objective)
        println("L2_LR_SGD test error: " + error)
        println("----------------------------")
      }
      case "L2_LR_GD" => {
        val l2lr = new L2_LR_GD(trainClass)
        val w = l2lr.train()
        val objective = l2lr.getObjective(w.toDenseVector, trainClass)
        val error = l2lr.testError(w, testClass.map(p => p.features), testClass.map(p => p.label))
        println("L2_LR_GD w: " + w)
        println("L2_LR_GD Objective value: " + objective)
        println("L2_LR_GD test error: " + error)
        println("----------------------------")
      }
      case "L1_LR_SGD" => {
        val l1lr = new L1_LR_SGD(trainClass)
        val w = l1lr.train()
        val objective = l1lr.getObjective(w.toDenseVector, trainClass)
        val error = l1lr.testError(w, testClass.map(p => p.features), testClass.map(p => p.label))
        println("L1_LR_SGD w: " + w)
        println("L1_LR_SGD Objective value: " + objective)
        println("L1_LR_SGD test error: " + error)
        println("----------------------------")
      }
      case "L1_LR_GD" => {
        val l1lr = new L1_LR_GD(trainClass)
        val w = l1lr.train()
        val objective = l1lr.getObjective(w.toDenseVector, trainClass)
        val error = l1lr.testError(w, testClass.map(p => p.features), testClass.map(p => p.label))
        println("L1_LR_GD w: " + w)
        println("L1_LR_GD Objective value: " + objective)
        println("L1_LR_GD test error: " + error)
        println("----------------------------")
      }
      case "L2_SVM_Cocoa" => {

        val lambda = 0.01
        val numRounds = 200 // number of outer iterations, called T in the paper
        val localIterFrac = 1.0 // fraction of local points to be processed per round, H = localIterFrac * n
        val beta = 1.0 // scaling parameter when combining the updates of the workers (1=averaging for CoCoA)
        val gamma = 1.0 // aggregation parameter for CoCoA+ (1=adding, 1/K=averaging)
        val debugIter = 10 // set to -1 to turn off debugging output
        val seed = 13 // set seed for debug purposes
        val n = trainClass.count().toInt
        var localIters = (localIterFrac * n / trainClass.partitions.size).toInt
        localIters = Math.max(localIters, 1)
        var chkptIter = 100
        val wInit = DenseVector.zeros[Double](trainClass.first().features.size)
        // set to solve hingeloss SVM
        val loss = distopt.utils.OptUtils.hingeLoss _
        val params = distopt.utils.Params(loss, n, wInit, numRounds, localIters, lambda, beta, gamma)
        val debug = distopt.utils.DebugParams(Utils.toCocoaFormat(testClass), debugIter, seed, chkptIter)

        val l2svm = new L2_SVM_COCOA(trainClass, params, debug, false)
        val w = l2svm.train()
        val objective = l2svm.getObjective(w.toDenseVector, trainClass)
        val error = l2svm.testError(w, testClass.map(p => p.features), testClass.map(p => p.label))
        println("L2_SVM_Cocoa w: " + w)
        println("L2_SVM_Cocoa Objective value: " + objective)
        println("L2_SVM_Cocoa test error: " + error)
        println("----------------------------")
      }
      case _ => println("The optimizer " + opt + " doest not exist")
    }
    }


    sc.stop()
  }

}
