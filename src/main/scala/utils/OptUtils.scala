package MLbenchmark.utils

import breeze.linalg.DenseVector
import breeze.linalg.SparseVector
import breeze.linalg.Vector
import org.apache.spark.SparkContext
import org.apache.spark.mllib.optimization._
import org.apache.spark.rdd.RDD
import org.apache.spark.mllib.linalg._


object OptUtils {

  // load data stored in LIBSVM format
  def loadLIBSVMData(sc: SparkContext, filename: String, numSplits: Int, numFeats: Int): RDD[LabeledPoint] = {

    // read in text file
    val data = sc.textFile(filename, numSplits).coalesce(numSplits) // note: coalesce can result in data being sent over the network. avoid this for large datasets
    val numEx = data.count()

    // find number of elements per partition
    val numParts = data.partitions.size
    val sizes = data.mapPartitionsWithSplit { case (i, lines) =>
      Iterator(i -> lines.length)
    }.collect().sortBy(_._1)
    val offsets = sizes.map(x => x._2).scanLeft(0)(_ + _).toArray

    // parse input
    data.mapPartitionsWithSplit { case (partition, lines) =>
      lines.zipWithIndex.flatMap { case (line, idx) =>

        // calculate index for line
        val index = offsets(partition) + idx

        if (index < numEx) {

          // parse label
          val parts = line.trim().split(' ')
          var label = -1
          if (parts(0).contains("+") || parts(0).toInt == 1)
            label = 1

          // parse features
          val featureArray = parts.slice(1, parts.length)
            .map(_.split(':')
          match { case Array(i, j) => (i.toInt - 1, j.toDouble)
          }).toArray
          val features = new SparseVector(featureArray.map(x => x._1), featureArray.map(x => x._2), numFeats)

          // create classification point
          Iterator(LabeledPoint(label, features))
        }
        else {
          Iterator()
        }
      }
    }
  }


  // calculate hinge loss
  def hingeLoss(point: LabeledPoint, w: Vector[Double]): Double = {
    val y = point.label
    val X = point.features
    return Math.max(1 - y * (X.dot(w)), 0.0)
  }

  def squareLoss(point: LabeledPoint, w: Vector[Double]): Double = {
    val y = point.label
    val X = point.features
    return Math.pow(1 - y * (X.dot(w)), 2)/2.0
  }


  // can be used to compute train or test error
  def computeAvgLoss(data: RDD[LabeledPoint], w: Vector[Double],lossType:Int): Double = {
    val n = data.count()
    lossType match{
      case 0 => return data.map(hingeLoss(_, w)).reduce(_ + _) / n
      case 1 => return data.map(squareLoss(_,w)).reduce(_ + _) / n
      case _ => return data.map(hingeLoss(_, w)).reduce(_ + _) / n
    }
  }

  def computeTotalLoss(data: RDD[LabeledPoint], w: Vector[Double]): Double = {
    return data.map(hingeLoss(_, w)).reduce(_ + _)
  }


  // Compute the primal objective function value.
  // Caution:just use for debugging purposes. this is an expensive operation, taking one full pass through the data
  def computePrimalObjective(data: RDD[LabeledPoint], w: Vector[Double], lambda: Double, lossType: Int): Double = {
    return computeAvgLoss(data, w, lossType) + (lambda * Math.pow(w.norm(2), 2))
  }



  def computeObjective(w: Vector[Double]): Double = {
    return Math.pow(w.norm(2),2)
  }


  // Compute the dual objective function value.
  // Caution:just use for debugging purposes. this is an expensive operation, taking one full pass through the data
  def computeDualObjective(data: RDD[LabeledPoint], w: Vector[Double], alpha : RDD[Vector[Double]], lambda: Double): Double = {
    val n = data.count()
    val sumAlpha = alpha.map(x => x.sum).reduce(_ + _)
    return (-lambda / 2 * Math.pow(w.norm(2), 2)) + (sumAlpha / n)
  }

  
  // Compute the duality gap value.
  // Caution:just use for debugging purposes. this is an expensive operation, taking one full pass through the data
  def computeDualityGap(data: RDD[LabeledPoint], w: Vector[Double], alpha: RDD[Vector[Double]], lambda: Double, lossType:Int): Double = {
    return (computePrimalObjective(data, w, lambda,lossType) - computeDualObjective(data, w, alpha, lambda))
  }


  // Compute the classification error.
  def computeClassificationError(data: RDD[LabeledPoint], w:Vector[Double]) : Double = {
    val n = data.count()
    return data.map(x => if((x.features).dot(w)*(x.label) > 0) 0.0 else 1.0).reduce(_ + _) / n
  }

  def computeMSE(data: RDD[LabeledPoint], w:Vector[Double]) : Double = {
    val n = data.count()
    return data.map(x => math.pow((x.features.dot(w) - x.label),2)).reduce(_+_)/n
  }


  // Print summary stats after the method has finished running (primal-dual).
//  def printSummaryStatsPrimalDual(algName: String, data: RDD[LabeledPoint], w: Vector[Double], alpha: RDD[Vector[Double]], lambda: Double, testData: RDD[LabeledPoint],lossType:Int) = {
//    var outString = algName + " has finished running. Summary Stats: "
//    val objVal = computePrimalObjective(data, w, lambda,lossType)
//    outString = outString + "\n Total Objective Value: " + objVal
//   val dualityGap = computeDualityGap(data, w, alpha, lambda)
//    outString = outString + "\n Duality Gap: " + dualityGap
//    if(testData!=null){
//      val testErr = computeClassificationError(testData, w)
//      outString = outString + "\n Test Error: " + testErr
//    }
//    println(outString + "\n")
//  }


  // Print summary stats after the method has finished running (primal only).
  def printSummaryStats(algName: String, data: RDD[LabeledPoint], w: Vector[Double], lambda: Double, testData: RDD[LabeledPoint],lossType:Int) =  {
    var outString = algName + " has finished running. Summary Stats: "
    val objVal = computePrimalObjective(data, w, lambda, lossType)
    outString = outString + "\n Total Objective Value: " + objVal
    if(testData!=null){
      val testErr = computeClassificationError(testData, w)
      outString = outString + "\n Test Error: " + testErr
    }
    println(outString + "\n")
  }

  def calOptimalVal(trainData: RDD[LabeledPoint],lossType:Int) = {
    val numFeatures = trainData.take(1)(0).features.size

    // Run training algorithm to build the model
    val numCorrections = 20
    val convergenceTol = 0
    var initialWeights = Vectors.dense(new Array[Double](numFeatures))
    val training = trainData.map(point => (point.label, Vectors.sparse(numFeatures, point.features.index, point.features.data)))

    lossType match {
      case 0 => {
        0.1718009546641592
      }
      case 1 => {
        val (weights, loss) = LBFGS.runLBFGS(
          training,
          new LeastSquaresGradient(),
          new SquaredL2Updater(),
          numCorrections,
          convergenceTol,
          100,
          0.001,
          initialWeights)
        val weightsVector = new DenseVector(weights.toArray)
        val optimalWeights = Vectors.dense(weights.toArray)
        val numExamples = training.count()
        CostFun.calculate(weights, training, new LeastSquaresGradient(), new SquaredL2Updater(), 0.001, numExamples)
        //    OptUtils.computePrimalObjective(trainData, weightsVector, 0.0005, lossType)
      }
      case 2 => {
//        val (weights, loss) = LoggingLbfgs.runLBFGS(
//          training,
//          new LeastSquaresGradient(),
//          new L1Updater(),
//          numCorrections,
//          convergenceTol,
//          400,
//          0.001,
//          initialWeights)
//        val weightsVector = new DenseVector(weights.toArray)
//        val optimalWeights = Vectors.dense(weights.toArray)
//        val numExamples = training.count()
//        println(CostFun.calculate(weights, training, new LeastSquaresGradient(), new L1Updater(), 0.001, numExamples))
        0.2187374179350034
      }
    }
  }
  
}
