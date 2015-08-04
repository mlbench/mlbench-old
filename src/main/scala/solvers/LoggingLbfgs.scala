package org.apache.spark.mllib.optimization

import java.io.{FileWriter, BufferedWriter, PrintWriter}
import scala.collection.mutable.ArrayBuffer

import breeze.linalg.{DenseVector => BDV}
import breeze.optimize.{CachedDiffFunction, DiffFunction, LBFGS => BreezeLBFGS}
import org.apache.spark.mllib.linalg.{DenseVector, Vector, Vectors}
import org.apache.spark.mllib.linalg.BLAS.axpy
import org.apache.spark.rdd.RDD
import org.apache.spark.Logging
import MLbenchmark.utils
/**
 * Created by xiyu on 15/7/24.
 */


object LoggingLbfgs extends Logging{

  def runLBFGS(data: RDD[(Double, Vector)],
               gradient: Gradient,
               updater: Updater,
               numCorrections: Int,
               convergenceTol: Double,
               maxNumIterations: Int,
               regParam: Double,
               initialWeights: Vector): (Vector, Array[Double]) = {

    val lossHistory = new ArrayBuffer[Double](maxNumIterations)

    var timeHistory = new ArrayBuffer[Double](maxNumIterations)

    val numExamples = data.count()

    val costFun =
      new CostFun(data, gradient, updater, regParam, numExamples)

    val lbfgs = new BreezeLBFGS[BDV[Double]](maxNumIterations, numCorrections, convergenceTol)

    val states =
      lbfgs.iterations(new CachedDiffFunction(costFun), initialWeights.toBreeze.toDenseVector)

    /**
     * NOTE: lossSum and loss is computed using the weights from the previous iteration
     * and regVal is the regularization value computed in the previous iteration as well.
     */
    var state = states.next()
    while(states.hasNext) {
      lossHistory.append(state.value)
      timeHistory.append(state.value)
      state = states.next()
    }
    lossHistory.append(state.value)
    val weights = Vectors.fromBreeze(state.x)

    logInfo("LBFGS.runLBFGS finished. Last 10 losses %s".format(
      lossHistory.takeRight(10).mkString(", ")))


    (weights, lossHistory.toArray)
  }
}

object CostFun {
  def calculate(w: Vector, data: RDD[(Double, Vector)], gradient: Gradient, updater: Updater,regParam: Double, numExamples: Long): Double = {
    var usedTime = System.nanoTime()

    val pw = new PrintWriter(new BufferedWriter(new FileWriter("output/time_" + "Lbfgs" + ".txt", true)))
    pw.println(usedTime / 1e6)
    pw.flush()
    pw.close
    // Have a local copy to avoid the serialization of CostFun object which is not serializable.
    val n = w.size
    val bcW = data.context.broadcast(w)
    val localGradient = gradient

    val (gradientSum, lossSum) = data.treeAggregate((Vectors.zeros(n), 0.0))(
      seqOp = (c, v) => (c, v) match {
        case ((grad, loss), (label, features)) =>
          val l = localGradient.compute(
            features, label, bcW.value, grad)
          (grad, loss + l)
      },
      combOp = (c1, c2) => (c1, c2) match {
        case ((grad1, loss1), (grad2, loss2)) =>
          axpy(1.0, grad2, grad1)
          (grad1, loss1 + loss2)
      })

    /**
     * regVal is sum of weight squares if it's L2 updater;
     * for other updater, the same logic is followed.
     */
    val regVal = updater.compute(w, Vectors.zeros(n), 0, 1, regParam)._2

    val objectiveValue = lossSum / numExamples + regVal

    objectiveValue
  }
}

class CostFun(data: RDD[(Double, Vector)],
                       gradient: Gradient,
                       updater: Updater,
                       regParam: Double,
                       numExamples: Long) extends DiffFunction[BDV[Double]] {

  override def calculate(weights: BDV[Double]): (Double, BDV[Double]) = {
    var usedTime = System.nanoTime()
    var pw = new PrintWriter(new BufferedWriter(new FileWriter("output/time_" +"Lbfgs"+".txt", true)))
    pw.println(usedTime/1e6)
    pw.flush()
    pw.close
    // Have a local copy to avoid the serialization of CostFun object which is not serializable.
    val w = Vectors.fromBreeze(weights)
    val n = w.size
    val bcW = data.context.broadcast(w)
    val localGradient = gradient

    val (gradientSum, lossSum) = data.treeAggregate((Vectors.zeros(n), 0.0))(
      seqOp = (c, v) => (c, v) match { case ((grad, loss), (label, features)) =>
        val l = localGradient.compute(
          features, label, bcW.value, grad)
        (grad, loss + l)
      },
      combOp = (c1, c2) => (c1, c2) match { case ((grad1, loss1), (grad2, loss2)) =>
        axpy(1.0, grad2, grad1)
        (grad1, loss1 + loss2)
      })

    /**
     * regVal is sum of weight squares if it's L2 updater;
     * for other updater, the same logic is followed.
     */
    val regVal = updater.compute(w, Vectors.zeros(n), 0, 1, regParam)._2

    val loss = lossSum / numExamples + regVal
    /**
     * It will return the gradient part of regularization using updater.
     *
     * Given the input parameters, the updater basically does the following,
     *
     * w' = w - thisIterStepSize * (gradient + regGradient(w))
     * Note that regGradient is function of w
     *
     * If we set gradient = 0, thisIterStepSize = 1, then
     *
     * regGradient(w) = w - w'
     *
     * TODO: We need to clean it up by separating the logic of regularization out
     *       from updater to regularizer.
     */
    // The following gradientTotal is actually the regularization part of gradient.
    // Will add the gradientSum computed from the data with weights in the next step.
    val gradientTotal = w.copy
    axpy(-1.0, updater.compute(w, Vectors.zeros(n), 1, 1, regParam)._1, gradientTotal)

    // gradientTotal = gradientSum / numExamples + gradientTotal
    axpy(1.0 / numExamples, gradientSum, gradientTotal)
    usedTime = System.nanoTime()
    pw = new PrintWriter(new BufferedWriter(new FileWriter("output/time_" +"Lbfgs"+".txt", true)))
    pw.println(usedTime/1e6)
    pw.flush()
    pw.close
    (loss, gradientTotal.toBreeze.asInstanceOf[BDV[Double]])
  }
}