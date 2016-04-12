package MLbenchmark.utils

import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.optimization._
import org.apache.spark.rdd.RDD
import breeze.linalg.{SparseVector, Vector}
import java.io._


// Labeled point with sparse features for classification or regression tasks
case class LabeledPoint(val label: Double, val features: SparseVector[Double])
case class DenseLabeledPoint(val label: Double, val features: org.apache.spark.mllib.linalg.Vector)


/** Algorithm Params
   * @param loss - the loss function l_i (assumed equal for all i)
   * @param n - number of data points
   * @param wInit - initial weight vector
   * @param numRounds - number of outer iterations (T in the paper)
   * @param localIters - number of inner localSDCA iterations (H in the paper)
   * @param lambda - the regularization parameter
   * @param beta - scaling parameter for CoCoA
   * @param gamma - aggregation parameter for CoCoA+ (gamma=1 for adding, gamma=1/K for averaging) 
   */
case class Params(
    loss: (LabeledPoint, Vector[Double]) => Double, 
    n: Int,
    wInit: Vector[Double], 
    numRounds: Int, 
    localIters: Int, 
    lambda: Double, 
    beta: Double,
    gamma: Double)


/** Debug Params
   * @param testData
   * @param debugIter
   * @param seed
   * @param chkptIter checkpointing the resulting RDDs from time to time, to ensure persistence and shorter dependencies
   */
case class DebugParams(
    testData: RDD[LabeledPoint],
    debugIter: Int,
    seed: Int,
    chkptIter: Int)

object DebugParamsML
{
    //lossType 0:hinge Loss, 1:square loss
    def calError(weights:Vector[Double], iterNum: Int, name: String,time:Long,lambda:Double,optimalVal:Double,lossType:Int, trainData: RDD[LabeledPoint], testData: RDD[LabeledPoint],fraction:Double) = {
      //val primalObjective = OptUtils.computePrimalObjective(trainData, weights, lambda, lossType)
      val weightsVector = Vectors.dense(weights.toArray)
      val training = trainData.map(point => (point.label, Vectors.sparse(point.features.length, point.features.index, point.features.data)))

      val primalObjective =
      lossType match {
        case 0 => CostFun.calculate (weightsVector, training, new HingeGradient, new SquaredL2Updater (), 0.001, trainData.count () )
        case 1 => CostFun.calculate (weightsVector, training, new LeastSquaresGradient, new SquaredL2Updater (), 0.001, trainData.count () )
        case 2 => CostFun.calculate (weightsVector, training, new LeastSquaresGradient, new L1Updater(), 0.001, trainData.count () )
        case _ => 0.0
      }

      val subObjective:Double = primalObjective - optimalVal

      var pw = new PrintWriter(new BufferedWriter(new FileWriter("output/Iter_" +name+".txt", true)))
      pw.println(iterNum)
      pw.flush()
      pw.close

      pw = new PrintWriter(new BufferedWriter(new FileWriter("output/primal_" +name+".txt", true)))
      pw.println(primalObjective)
      pw.flush()
      pw.close

      pw = new PrintWriter(new BufferedWriter(new FileWriter("output/sub_" +name+".txt", true)))
      pw.println(subObjective)
      pw.flush()
      pw.close

      pw = new PrintWriter(new BufferedWriter(new FileWriter("output/time_" +name+".txt", true)))
      pw.println(time)
      pw.flush()
      pw.close
      if(math.floor(iterNum.toDouble * fraction)-math.floor((iterNum-1).toDouble*fraction)==1) {
        pw = new PrintWriter(new BufferedWriter(new FileWriter("output/pass_primal_" + name + ".txt", true)))
        pw.println(primalObjective)
        pw.flush()
        pw.close
        pw = new PrintWriter(new BufferedWriter(new FileWriter("output/pass_sub_" + name + ".txt", true)))
        pw.println(subObjective)
        pw.flush()
        pw.close
      }
      println("PrimalObjective: "  + primalObjective)
      println("subObjective: " + subObjective)
      println("iterations: " + iterNum)
      println("time: " + time + "ms")
    }
}
