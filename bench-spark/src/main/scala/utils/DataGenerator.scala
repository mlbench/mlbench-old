// 
// Partially from spark-perf package
// 
package utils.DataGenerator

import scala.collection.mutable

import org.apache.spark.mllib.linalg.{Vectors, Vector}
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.mllib.random._
import org.apache.spark.mllib.recommendation.Rating
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.configuration.{Algo, FeatureType}
import org.apache.spark.mllib.tree.model.{Split, DecisionTreeModel, Node, Predict}
import org.apache.spark.rdd.{PairRDDFunctions, RDD}
import org.apache.spark.SparkContext

object DataGenerator {

  def generateLabeledPoints(
    sc: SparkContext,
    numRows: Long,
    numCols: Int,
    intercept: Double,
    labelNoise: Double,
    numPartitions: Int,
    seed: Long = System.currentTimeMillis(),
    problem: String = ""): RDD[LabeledPoint] = {

    RandomRDDs.randomRDD(sc, new LinearDataGenerator(numCols,intercept, seed, labelNoise, problem),
      numRows, numPartitions, seed)
  }

  class LinearDataGenerator(
    val numFeatures: Int,
    val intercept: Double,
    val seed: Long,
    val labelNoise: Double,
    val problem: String = "",
    val sparsity: Double = 1.0) extends RandomDataGenerator[LabeledPoint] {

    private val rng = new java.util.Random(seed)
    private val weights = Array.fill(numFeatures)(rng.nextDouble())
    private val nnz: Int = math.ceil(numFeatures*sparsity).toInt

    override def nextValue(): LabeledPoint = {
      val x = Array.fill[Double](nnz)(2*rng.nextDouble()-1)

      val y = weights.zip(x).map(p => p._1 * p._2).sum + intercept + labelNoise*rng.nextGaussian()
      val yD =
        if (problem == "SVM"){
          if (y < 0.0) 0.0 else 1.0
        } else{
          y
        }

      LabeledPoint(yD, Vectors.dense(x))
    }

    override def setSeed(seed: Long) {
      rng.setSeed(seed)
    }

    override def copy(): LinearDataGenerator =
      new LinearDataGenerator(numFeatures, intercept, seed, labelNoise, problem, sparsity)
    }
}
