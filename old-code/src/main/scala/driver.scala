package MLbenchmark

import _root_.solvers.CoCoA
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.{SparkContext, SparkConf}
import MLbenchmark.solvers._
import MLbenchmark.utils._
import breeze.linalg.DenseVector
import org.apache.spark.mllib.util.{LinearDataGenerator, MLUtils}
import org.apache.spark.sql._

object driver {
	def main(args: Array[String]) {

    val options =  args.map { arg =>
      arg.dropWhile(_ == '-').split('=') match {
        case Array(opt, v) => (opt -> v)
        case Array(opt) => (opt -> "true")
        case _ => throw new IllegalArgumentException("Invalid argument: " + arg)
      }
    }.toMap

    // read in inputs
    val master = options.getOrElse("master", "local[4]")
    val trainFile = options.getOrElse("trainFile", "")
    val numFeatures = options.getOrElse("numFeatures", "0").toInt
    val numSplits = options.getOrElse("numSplits","1").toInt
    val chkptDir = options.getOrElse("chkptDir","");
    var chkptIter = options.getOrElse("chkptIter","100").toInt
    val testFile = options.getOrElse("testFile", "")
    val justCoCoA = options.getOrElse("justCoCoA", "true").toBoolean // set to false to compare different methods

    // algorithm-specific inputs
    val lambda = options.getOrElse("lambda", "0.01").toDouble // regularization parameter
    val numRounds = options.getOrElse("numRounds", "200").toInt // number of outer iterations, called T in the paper
    val localIterFrac = options.getOrElse("localIterFrac","1.0").toDouble; // fraction of local points to be processed per round, H = localIterFrac * n
    val beta = options.getOrElse("beta","1.0").toDouble;  // scaling parameter when combining the updates of the workers (1=averaging for CoCoA)
    val gamma = options.getOrElse("gamma","1.0").toDouble;  // aggregation parameter for CoCoA+ (1=adding, 1/K=averaging)
    val debugIter = options.getOrElse("debugIter","10").toInt // set to -1 to turn off debugging output
    val seed = options.getOrElse("seed","0").toInt // set seed for debug purposes

    // print out inputs
    println("master:       " + master);          println("trainFile:    " + trainFile);
    println("numFeatures:  " + numFeatures);     println("numSplits:    " + numSplits);
    println("chkptDir:     " + chkptDir);        println("chkptIter     " + chkptIter);       
    println("testfile:     " + testFile);        println("justCoCoA     " + justCoCoA);       
    println("lambda:       " + lambda);          println("numRounds:    " + numRounds);       
    println("localIterFrac:" + localIterFrac);   println("beta          " + beta);     
    println("gamma         " + beta);            println("debugIter     " + debugIter);       
    println("seed          " + seed);


    // start spark context
    val conf = new SparkConf().setMaster(master)
      .setAppName("demoCoCoA")
      .setJars(SparkContext.jarOfObject(this).toSeq)
    val sc = new SparkContext(conf)
    if (chkptDir != "") {
      sc.setCheckpointDir(chkptDir)
    } else {
      chkptIter = numRounds + 1
    }


    // read in data
    val train_data = OptUtils.loadLIBSVMData(sc,trainFile,numSplits,numFeatures)
    val train_data_dense = OptUtils.loadDenseLIBSVMData(sc,trainFile,numSplits,numFeatures)
    //val optimalVal = OptUtils.calOptimalVal(train_data,2)
    val optimalVal = 0.0
    val n = train_data.count().toInt // number of data examples
    val test_data = {
      if (testFile != ""){ OptUtils.loadLIBSVMData(sc,testFile,numSplits,numFeatures).cache() }
      else { null }
    }


    //create Data Frame which is used in OWLQN
    val sqContext = new SQLContext(sc)
    val dataset = sqContext.createDataFrame(train_data_dense)

    // compute H, # of local iterations
    var localIters = (localIterFrac * n / train_data.partitions.size).toInt
    localIters = Math.max(localIters,1)

    // for the primal-dual algorithms to run correctly, the initial primal vector has to be zero 
    // (corresponding to dual alphas being zero)
    val wInit = DenseVector.zeros[Double](numFeatures)

    // set to solve hingeloss SVM
    val loss = OptUtils.hingeLoss _
    val params = Params(loss, n, wInit, numRounds, localIters, lambda, beta, gamma)
    val debug = DebugParams(test_data, 2, seed, chkptIter)

    //MLlib sgd
    SGD.run_SGD(
        trainData = train_data,
        testData = test_data,
        maxIter = numRounds,
        chkptIter = 2,
        optimalVal = optimalVal,
        lossType = 2)

    //MLlib l-bfgs
    Lbfgs.run_LBFGS(
      train_data,
      test_data,
      dataset,
      numRounds,
      2,
      optimalVal,
      2)


//  	val (finalwCoCoA, finalalphaCoCoA) =
//    CoCoA.runCoCoA(
//      train_data,
//      params,
//      debug,
//      plus = false,
//      optimalVal,
//      localIterFrac,
//      lossType = 0)
  	sc.stop
	}
}