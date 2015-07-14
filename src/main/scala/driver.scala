package MLbenchmark
import org.apache.spark.{SparkContext, SparkConf}
import MLbenchmark.solvers._
import MLbenchmark.utils._
import breeze.linalg.DenseVector
import org.apache.spark.mllib.util.MLUtils

object driver {
	// def main(args: Array[String]) {
	// 	val sc = new SparkContext("local","MLbenchmark","/Users/mac/Desktop/summer_project/spark-1.4.0/bin/",List("target/scala-2.10/driver_2.10-1.0.jar"))

	// 	val train_file = "data/small_train.dat" 
	// 	val test_file = "data/small_test.dat"
	// 	val numFeatures = 9947
	// 	val debugIter = 10
	// 	val seed = 0
	// 	val chkptIter = 201
	// 	//read in
	// 	val train_data = OptUtils.loadLIBSVMData(sc, train_file, 4, numFeatures).cache()
	// 	val test_data = OptUtils.loadLIBSVMData(sc, test_file, 4, numFeatures).cache()

	// 	val LocalIterFrac = 0.1
	// 	val loss = OptUtils.hingeLoss _
	// 	val n = train_data.count().toInt
	// 	val wInit = DenseVector.zeros[Double](numFeatures)

	// 	val numRounds = 200
	// 	val localIters =  (LocalIterFrac * n/ train_data.partitions.size).toInt
	// 	val lambda = 0.001 
	// 	val beta = 1.0
	// 	val gamma = 1.0
	// 	val params = Params(loss, n, wInit, numRounds, localIters, lambda, beta, gamma)
	// 	val debug = DebugParams(test_data, debugIter, seed, chkptIter)
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
    val train_data = OptUtils.loadLIBSVMData(sc,trainFile,numSplits,numFeatures).cache()

    val n = train_data.count().toInt // number of data examples
    val test_data = {
      if (testFile != ""){ OptUtils.loadLIBSVMData(sc,testFile,numSplits,numFeatures).cache() }
      else { null }
    }

    // compute H, # of local iterations
    var localIters = (localIterFrac * n / train_data.partitions.size).toInt
    localIters = Math.max(localIters,1)

    // for the primal-dual algorithms to run correctly, the initial primal vector has to be zero 
    // (corresponding to dual alphas being zero)
    val wInit = DenseVector.zeros[Double](numFeatures)

    // set to solve hingeloss SVM
    val loss = OptUtils.hingeLoss _
    val params = Params(loss, n, wInit, numRounds, localIters, lambda, beta, gamma)
    val debug = DebugParams(test_data, debugIter, seed, chkptIter)
     for(iter <- 1 to 5){
         SGD.run_SGD(train_data, test_data,iter*1000)
     }
     for(iter <- 1 to 20){
         Lbfgs.run_LBFGS(train_data, test_data, iter * 10)
     }
	val (finalwCoCoA, finalalphaCoCoA) = CoCoA_ridge.runCoCoA(train_data, params, debug, plus = false)
	OptUtils.printSummaryStatsPrimalDual("CoCoA", train_data, finalwCoCoA, finalalphaCoCoA, lambda, test_data)
	sc.stop
	}
}