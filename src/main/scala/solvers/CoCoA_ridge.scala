package MLbenchmark.solvers

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import MLbenchmark.utils._
import breeze.linalg.{Vector, NumericOps, DenseVector, SparseVector}


object CoCoA_ridge {

  /**
   * CoCoA/CoCoA+ - Communication-efficient distributed dual Coordinate Ascent.
   * Using LocalSDCA as the local dual method. Here implemented for standard 
   * hinge-loss SVM. For other objectives, adjust localSDCA accordingly.
   * 
   * @param data RDD of all data examples
   * @param params Algorithmic parameters
   * @param debug  Systems/debugging parameters
   * @param plus Whether to use the CoCoA+ framework (plus=true) or CoCoA (plus=false)
   * @return
   */
  def runCoCoA(
    data: RDD[LabeledPoint],
    params: Params,
    debug: DebugParams,
    plus: Boolean) : (Vector[Double], RDD[Vector[Double]]) = {
    
    val parts = data.partitions.size 	// number of partitions of the data, K in the paper
    val alg = if (plus) "CoCoA+" else "CoCoA"
    println("\nRunning "+alg+" on "+params.n+" data examples, distributed over "+parts+" workers")
    
    // initialize alpha, w
    var alphaVars = data.map(x => 0.0).cache()
    var alpha = alphaVars.mapPartitions(x => Iterator(Vector(x.toArray)))
    var dataArr = data.mapPartitions(x => Iterator(x.toArray))
    var w = params.wInit.copy
    var scaling = if (plus) params.gamma else params.beta/parts

    for(t <- 1 to 200) {

      // zip alpha with data
      val zipData = alpha.zip(dataArr)

      // find updates to alpha, w
      val updates = zipData.mapPartitions(partitionUpdate(_, w, params.localIters, params.lambda, params.n, scaling, debug.seed + t, plus, parts * params.gamma), preservesPartitioning = true).persist()
      alpha = updates.map(kv => kv._2)
      val primalUpdates = updates.map(kv => kv._1).reduce(_ + _)
      w += (primalUpdates * scaling)

      // optionally calculate errors
      if (debug.debugIter > 0 && t % debug.debugIter == 0) {
        println("Iteration: " + t)
        println("primal objective: " + OptUtils.computePrimalObjective(data, w, params.lambda))
        println("primal-dual gap: " + OptUtils.computeDualityGap(data, w, alpha, params.lambda))
        val objectFunctionCheck = OptUtils.computeObjective(w)
//        println("object function" + objectFunctionCheck)
        if (debug.testData != null) { println("Classsification error: " + OptUtils.computeClassificationError(debug.testData, w)) }
      }

      // optionally checkpoint RDDs
      if(t % debug.chkptIter == 0){
        zipData.checkpoint()
        alpha.checkpoint()
      }
    }

    return (w, alpha)
  }


  /**
   * Performs one round of local updates using a given local dual algorithm, 
   * here locaSDCA. Will perform localIters many updates per worker.
   *
   * @param zipData
   * @param winit
   * @param localIters
   * @param lambda
   * @param n
   * @param scaling This is either gamma for CoCoA+ or beta/K for CoCoA
   * @param seed
   * @param plus
   * @param sigma sigma' in the CoCoA+ paper
   * @return
   */
  private def partitionUpdate(
    zipData: Iterator[(Vector[Double],Array[LabeledPoint])],//((Int, Double), SparseClassificationPoint)],
    wInit: Vector[Double], 
    localIters: Int, 
    lambda: Double, 
    n: Int, 
    scaling: Double,
    seed: Int,
    plus: Boolean,
    sigma: Double): Iterator[(Vector[Double], Vector[Double])] = {

    val zipPair = zipData.next()
    val localData = zipPair._2
    var alpha = zipPair._1
    val alphaOld = alpha.copy

    val (deltaAlpha, deltaW) = localSDCA(localData, wInit, localIters, lambda, n, alpha, alphaOld, seed, plus, sigma)
    alpha = alphaOld + (deltaAlpha * scaling)

    return Iterator((deltaW, alpha))
  }


  /**
   * This is an implementation of LocalDualMethod, here LocalSDCA (coordinate ascent),
   * with taking the information of the other workers into account, by respecting the
   * shared wInit vector.
   * Here we perform coordinate updates for the SVM dual objective (hinge loss).
   *
   * Note that SDCA for hinge-loss is equivalent to LibLinear, where using the
   * regularization parameter  C = 1.0/(lambda*numExamples), and re-scaling
   * the alpha variables with 1/C.
   *
   * @param localData The local data examples
   * @param wInit
   * @param localIters Number of local coordinates to update
   * @param lambda
   * @param n Global number of points (needed for the primal-dual correspondence)
   * @param alpha
   * @param alphaOld
   * @param seed
   * @param plus
   * @param sigma
   * @return (deltaAlpha, deltaW) Summarizing the performed local changes
   */
  def localSDCA(
    localData: Array[LabeledPoint],
    wInit: Vector[Double], 
    localIters: Int, 
    lambda: Double, 
    n: Int,
    alpha: Vector[Double], 
    alphaOld: Vector[Double],
    seed: Int,
    plus: Boolean,
    sigma: Double): (Vector[Double], Vector[Double]) = {
    
    var w = wInit
    val nLocal = localData.length
    var r = new scala.util.Random(seed)
    var deltaW = DenseVector.zeros[Double](wInit.length)

    // perform local udpates
    for (i <- 1 to localIters) {

      // randomly select a local example
      val idx = r.nextInt(nLocal)
      val currPt = localData(idx)
      var y = currPt.label
      val x = currPt.features

      //calculate deltaAlpha
      val xnorm = Math.pow(x.norm(2), 2)
      val newAlpha = (y - x.dot(w) - 0.5 * alpha(idx))/(0.5 + xnorm/(lambda * n)) + alpha(idx)

      // update primal and dual variables
      val update = x* (newAlpha - alpha(idx)) / (lambda * n)
      w += update
      deltaW += update
      alpha(idx) = newAlpha
    }

    val deltaAlpha = alpha - alphaOld
    return (deltaAlpha, deltaW)
  }

}
