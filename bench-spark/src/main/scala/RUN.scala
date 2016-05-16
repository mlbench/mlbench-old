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

    val output = new File(workingDir + "res.out")
    val bw = new BufferedWriter(new FileWriter(output))
    //Run all optimisers given in the args
    optimizers.foreach { opt => opt match {
      case "Elastic_ProxCocoa" => {
        val l1net = new Elastic_ProxCOCOA(train, test)
        val w = l1net.fit()
        bw.write("Elastic_ProxCocoa: " + "lambda: " +
          l1net.regularizer.lambda + " alpha: " + l1net.regularizer.asInstanceOf[ElasticNet].alpha + " elapsed: " +
          l1net.elapsed.get / 1000 / 1000 + "ms " + w.toDenseVector)
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
        val l1lasso = new L1_Lasso_ProxCocoa(train, test)
        val w = l1lasso.fit()
        bw.write("L1_Lasso_ProxCocoa: " + "lambda: " +
          l1lasso.regularizer.lambda + " elapsed: " + l1lasso.elapsed.get / 1000 / 1000 + "ms " + w.toDenseVector )
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
        val l1lasso = new L1_Lasso_GD(train)
        val w = l1lasso.fit()
        bw.write("L1_Lasso_GD: " + "lambda: " +
          l1lasso.regularizer.lambda + " elapsed: " + l1lasso.elapsed.get / 1000 / 1000 + "ms " + w.toDenseVector )
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
        val l1lasso = new L1_Lasso_SGD(train)
        val w = l1lasso.fit()
        bw.write("L1_Lasso_SGD: " + "lambda: " +
          l1lasso.regularizer.lambda + " elapsed: " + l1lasso.elapsed.get / 1000 / 1000 + "ms " + w.toDenseVector )
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
        val l2svm = new L2_SVM_SGD(train)
        val w = l2svm.train()
        bw.write("L2_SVM_SGD: " + "lambda: " +
          l2svm.regularizer.lambda + " elapsed: " + l2svm.elapsed.get / 1000 / 1000 + "ms " + w.toDenseVector )
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
        val l2svm = new L2_SVM_GD(train)
        val w = l2svm.train()
        bw.write("L2_SVM_GD: " + "lambda: " +
          l2svm.regularizer.lambda + " elapsed: " + l2svm.elapsed.get / 1000 / 1000 + "ms " + w.toDenseVector )
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
        val l2lr = new L2_LR_SGD(train)
        val w = l2lr.train()
        bw.write("L2_LR_SGD: " + "lambda: " +
          l2lr.regularizer.lambda + " elapsed: " + l2lr.elapsed.get / 1000 / 1000 + "ms " + w.toDenseVector )
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
        val l2lr = new L2_LR_GD(train)
        val w = l2lr.train()
        bw.write("L2_LR_GD: " + "lambda: " +
          l2lr.regularizer.lambda + " elapsed: " + l2lr.elapsed.get / 1000 / 1000 + "ms " + w.toDenseVector )
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
        val l1lr = new L1_LR_SGD(train)
        val w = l1lr.train()
        bw.write("L1_LR_SGD: " + "lambda: " +
          l1lr.regularizer.lambda + " elapsed: " + l1lr.elapsed.get / 1000 / 1000 + "ms " + w.toDenseVector )
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
        val l1lr = new L1_LR_GD(train)
        val w = l1lr.train()
        bw.write("L1_LR_GD: " + "lambda: " +
          l1lr.regularizer.lambda + " elapsed: " + l1lr.elapsed.get / 1000 / 1000 + "ms " + w.toDenseVector )
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
        val l2svm = new L2_SVM_COCOA(train, test, false)
        val w = l2svm.train()
        bw.write("L2_SVM_Cocoa: " + "lambda: " +
          l2svm.regularizer.lambda + " elapsed: " + l2svm.elapsed.get / 1000 / 1000 + "ms " + w.toDenseVector )
        bw.newLine()
        val objective = l2svm.getObjective(w.toDenseVector, train)
        val error = l2svm.testError(w, test.map(p => p.features), test.map(p => p.label))
        println("Training took: " + l2svm.elapsed.get / 1000 / 1000 + "ms")
        println("L2_SVM_Cocoa w: " + w)
        println("L2_SVM_Cocoa Objective value: " + objective)
        println("L2_SVM_Cocoa test error: " + error)
        println("----------------------------")
      }
      case "Mllib_Lasso_SGD" => {
        val lasso = new Mllib_Lasso_SGD(train)
        val w = lasso.fit()
        bw.write("Mllib_Lasso_SGD: " + "lambda: " +
          lasso.regularizer.lambda + " elapsed: " + lasso.elapsed.get / 1000 / 1000 + "ms " + w.toDenseVector )
        bw.newLine()
        val objective = lasso.getObjective(w.toDenseVector, train)
        val error = lasso.testError(w, test.map(p => p.features), test.map(p => p.label))
        println("Training took: " + lasso.elapsed.get / 1000 / 1000 + "ms")
        println("Mllib_Lasso_SGD w: " + w)
        println("Mllib_Lasso_SGD Objective value: " + objective)
        println("Mllib_Lasso_SGD test error: " + error)
        println("----------------------------")
      }
      case "Mllib_Lasso_GD" => {
        val lasso = new Mllib_Lasso_GD(train)
        val w = lasso.fit()
        bw.write("Mllib_Lasso_GD: " + "lambda: " +
          lasso.regularizer.lambda + " elapsed: " + lasso.elapsed.get / 1000 / 1000 + "ms " + w.toDenseVector )
        bw.newLine()
        val objective = lasso.getObjective(w.toDenseVector, train)
        val error = lasso.testError(w, test.map(p => p.features), test.map(p => p.label))
        println("Training took: " + lasso.elapsed.get / 1000 / 1000 + "ms")
        println("Mllib_Lasso_GD w: " + w)
        println("Mllib_Lasso_GD Objective value: " + objective)
        println("Mllib_Lasso_GD test error: " + error)
        println("----------------------------")
      }
      case "Mllib_L2_LR_LBFGS" => {
        val lbfgs = new Mllib_L2_LR_LBFGS(train)
        val w = lbfgs.train()
        bw.write("Mllib_L2_LR_LBFGS: " + "lambda: " +
          lbfgs.regularizer.lambda + " elapsed: " + lbfgs.elapsed.get / 1000 / 1000 + "ms " + w.toDenseVector )
        bw.write("Mllib_L2_LR_LBFGS: " + w + " elapsed: " + lbfgs.elapsed.get / 1000 / 1000 + "ms " + "lambda: " +
          lbfgs.regularizer.lambda)
        bw.newLine()
        val objective = lbfgs.getObjective(w.toDenseVector, train)
        val error = lbfgs.testError(w, test.map(p => p.features), test.map(p => p.label))
        println("Training took: " + lbfgs.elapsed.get / 1000 / 1000 + "ms")
        println("Mllib_L2_LR_LBFGS w: " + w)
        println("Mllib_L2_LR_LBFGS Objective value: " + objective)
        println("Mllib_L2_LR_LBFGS test error: " + error)
        println("----------------------------")
      }
      case "Mllib_L1_LR_LBFGS" => {
        val lbfgs = new Mllib_L1_LR_LBFGS(train)
        val w = lbfgs.train()
        bw.write("Mllib_L1_LR_LBFGS: " + "lambda: " +
          lbfgs.regularizer.lambda + " elapsed: " + lbfgs.elapsed.get / 1000 / 1000 + "ms " + w.toDenseVector )
        bw.newLine()
        val objective = lbfgs.getObjective(w.toDenseVector, train)
        val error = lbfgs.testError(w, test.map(p => p.features), test.map(p => p.label))
        println("Training took: " + lbfgs.elapsed.get / 1000 / 1000 + "ms")
        println("Mllib_L1_LR_LBFGS w: " + w)
        println("Mllib_L1_LR_LBFGS Objective value: " + objective)
        println("Mllib_L1_LR_LBFGS test error: " + error)
        println("----------------------------")
      }

      case "Mllib_L1_LR_GD" => {
        val l1lr = new Mllib_L1_LR_GD(train)
        val w = l1lr.train()
        bw.write("Mllib_L1_LR_GD: " + "lambda: " +
          l1lr.regularizer.lambda + " elapsed: " + l1lr.elapsed.get / 1000 / 1000 + "ms " + w.toDenseVector )
        bw.newLine()
        val objective = l1lr.getObjective(w.toDenseVector, train)
        val error = l1lr.testError(w, test.map(p => p.features), test.map(p => p.label))
        println("Training took: " + l1lr.elapsed.get / 1000 / 1000 + "ms")
        println("Mllib_L1_LR_GD w: " + w)
        println("Mllib_L1_LR_GD Objective value: " + objective)
        println("Mllib_L1_LR_GD test error: " + error)
        println("----------------------------")
      }
      case "Mllib_L1_LR_SGD" => {
        val l1lr = new Mllib_L1_LR_SGD(train)
        val w = l1lr.train()
        bw.write("Mllib_L1_LR_SGD: " + "lambda: " +
          l1lr.regularizer.lambda + " elapsed: " + l1lr.elapsed.get / 1000 / 1000 + "ms " + w.toDenseVector )
        bw.newLine()
        val objective = l1lr.getObjective(w.toDenseVector, train)
        val error = l1lr.testError(w, test.map(p => p.features), test.map(p => p.label))
        println("Training took: " + l1lr.elapsed.get / 1000 / 1000 + "ms")
        println("Mllib_L1_LR_SGD w: " + w)
        println("Mllib_L1_LR_SGD Objective value: " + objective)
        println("Mllib_L1_LR_SGD test error: " + error)
        println("----------------------------")
      }
      case "Mllib_L2_LR_SGD" => {
        val l1lr = new Mllib_L2_LR_SGD(train)
        val w = l1lr.train()
        bw.write("Mllib_L2_LR_SGD: " + "lambda: " +
          l1lr.regularizer.lambda + " elapsed: " + l1lr.elapsed.get / 1000 / 1000 + "ms " + w.toDenseVector )
        bw.newLine()
        val objective = l1lr.getObjective(w.toDenseVector, train)
        val error = l1lr.testError(w, test.map(p => p.features), test.map(p => p.label))
        println("Training took: " + l1lr.elapsed.get / 1000 / 1000 + "ms")
        println("Mllib_L2_LR_SGD w: " + w)
        println("Mllib_L2_LR_SGD Objective value: " + objective)
        println("Mllib_L2_LR_SGD test error: " + error)
        println("----------------------------")
      }
      case "Mllib_L2_LR_GD" => {
        val l1lr = new Mllib_L2_LR_GD(train)
        val w = l1lr.train()
        bw.write("Mllib_L2_LR_GD: " + "lambda: " +
          l1lr.regularizer.lambda + " elapsed: " + l1lr.elapsed.get / 1000 / 1000 + "ms " + w.toDenseVector )
        bw.newLine()
        val objective = l1lr.getObjective(w.toDenseVector, train)
        val error = l1lr.testError(w, test.map(p => p.features), test.map(p => p.label))
        println("Training took: " + l1lr.elapsed.get / 1000 / 1000 + "ms")
        println("Mllib_L2_LR_GD w: " + w)
        println("Mllib_L2_LR_GD Objective value: " + objective)
        println("Mllib_L2_LR_GD test error: " + error)
        println("----------------------------")
      }
      case "Mllib_L2_SVM_SGD" => {
        val l1lr = new Mllib_L2_SVM_SGD(train)
        val w = l1lr.train()
        bw.write("Mllib_L2_SVM_SGD: " + "lambda: " +
          l1lr.regularizer.lambda + " elapsed: " + l1lr.elapsed.get / 1000 / 1000 + "ms " + w.toDenseVector )
        bw.newLine()
        val objective = l1lr.getObjective(w.toDenseVector, train)
        val error = l1lr.testError(w, test.map(p => p.features), test.map(p => p.label))
        println("Training took: " + l1lr.elapsed.get / 1000 / 1000 + "ms")
        println("Mllib_L2_SVM_SGD w: " + w)
        println("Mllib_L2_SVM_SGD Objective value: " + objective)
        println("Mllib_L2_SVM_SGD test error: " + error)
        println("----------------------------")
      }

      case "Mllib_L2_SVM_GD" => {
        val l1lr = new Mllib_L2_SVM_GD(train)
        val w = l1lr.train()
        bw.write("Mllib_L2_SVM_GD: " + "lambda: " +
          l1lr.regularizer.lambda + " elapsed: " + l1lr.elapsed.get / 1000 / 1000 + "ms " + w.toDenseVector )
        bw.newLine()
        val objective = l1lr.getObjective(w.toDenseVector, train)
        val error = l1lr.testError(w, test.map(p => p.features), test.map(p => p.label))
        println("Training took: " + l1lr.elapsed.get / 1000 / 1000 + "ms")
        println("Mllib_L2_SVM_GD w: " + w)
        println("Mllib_L2_SVM_GD Objective value: " + objective)
        println("Mllib_L2_SVM_GD test error: " + error)
        println("----------------------------")
      }
      case _ => println("The optimizer " + opt + " doest not exist")
    }
    }
    bw.close()
    sc.stop()
  }
}
