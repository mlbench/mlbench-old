/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.ml.regression

import java.io.{FileWriter, BufferedWriter, PrintWriter}

import org.apache.spark.mllib.optimization.{L1Updater, LeastSquaresGradient, CostFun}

import scala.collection.mutable

import breeze.linalg.{DenseVector => BDV, norm => brzNorm}
import breeze.optimize.{CachedDiffFunction, DiffFunction, LBFGS => BreezeLBFGS, OWLQN => BreezeOWLQN}

import org.apache.spark.Logging
import org.apache.spark.annotation.Experimental
import org.apache.spark.ml.PredictorParams
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.{HasElasticNetParam, HasMaxIter, HasRegParam, HasTol}
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.linalg.BLAS._
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.stat.MultivariateOnlineSummarizer
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame
import org.apache.spark.storage.StorageLevel
import org.apache.spark.util.StatCounter

/**
 * Params for linear regression.
 */
private[regression] trait LinearRegressionParams extends PredictorParams
with HasRegParam with HasElasticNetParam with HasMaxIter with HasTol

/**
 * :: Experimental ::
 * Linear regression.
 *
 * The learning objective is to minimize the squared error, with regularization.
 * The specific squared error loss function used is:
 *   L = 1/2n ||A weights - y||^2^
 *
 * This support multiple types of regularization:
 *  - none (a.k.a. ordinary least squares)
 *  - L2 (ridge regression)
 *  - L1 (Lasso)
 *  - L2 + L1 (elastic net)
 */
@Experimental
class LoggingLinearRegression(override val uid: String)
  extends Regressor[Vector, LinearRegression, LinearRegressionModel]
  with LinearRegressionParams with Logging {

  def this() = this(Identifiable.randomUID("linReg"))

  /**
   * Set the regularization parameter.
   * Default is 0.0.
   * @group setParam
   */
  def setRegParam(value: Double): this.type = set(regParam, value)
  setDefault(regParam -> 0.0)

  /**
   * Set the ElasticNet mixing parameter.
   * For alpha = 0, the penalty is an L2 penalty. For alpha = 1, it is an L1 penalty.
   * For 0 < alpha < 1, the penalty is a combination of L1 and L2.
   * Default is 0.0 which is an L2 penalty.
   * @group setParam
   */
  def setElasticNetParam(value: Double): this.type = set(elasticNetParam, value)
  setDefault(elasticNetParam -> 0.0)

  /**
   * Set the maximum number of iterations.
   * Default is 100.
   * @group setParam
   */
  def setMaxIter(value: Int): this.type = set(maxIter, value)
  setDefault(maxIter -> 100)

  /**
   * Set the convergence tolerance of iterations.
   * Smaller value will lead to higher accuracy with the cost of more iterations.
   * Default is 1E-6.
   * @group setParam
   */
  def setTol(value: Double): this.type = set(tol, value)
  setDefault(tol -> 1E-6)

  override protected def train(dataset: DataFrame): LinearRegressionModel = {
    // Extract columns from data.  If dataset is persisted, do not persist instances.
    val instances = extractLabeledPoints(dataset).map {
      case LabeledPoint(label: Double, features: Vector) => (label, features)
    }
    val handlePersistence = dataset.rdd.getStorageLevel == StorageLevel.NONE
    if (handlePersistence) instances.persist(StorageLevel.MEMORY_AND_DISK)

    val (summarizer, statCounter) = instances.treeAggregate(
      (new MultivariateOnlineSummarizer, new StatCounter))(
        seqOp = (c, v) => (c, v) match {
          case ((summarizer: MultivariateOnlineSummarizer, statCounter: StatCounter),
          (label: Double, features: Vector)) =>
            (summarizer.add(features), statCounter.merge(label))
        },
        combOp = (c1, c2) => (c1, c2) match {
          case ((summarizer1: MultivariateOnlineSummarizer, statCounter1: StatCounter),
          (summarizer2: MultivariateOnlineSummarizer, statCounter2: StatCounter)) =>
            (summarizer1.merge(summarizer2), statCounter1.merge(statCounter2))
        })

    val numFeatures = summarizer.mean.size
    val yMean = statCounter.mean
    val yStd = math.sqrt(statCounter.variance)

    // If the yStd is zero, then the intercept is yMean with zero weights;
    // as a result, training is not needed.
    if (yStd == 0.0) {
      logWarning(s"The standard deviation of the label is zero, so the weights will be zeros " +
        s"and the intercept will be the mean of the label; as a result, training is not needed.")
      if (handlePersistence) instances.unpersist()
      return new LinearRegressionModel(uid, Vectors.sparse(numFeatures, Seq()), yMean)
    }

    val featuresMean = summarizer.mean.toArray
    val featuresStd = summarizer.variance.toArray.map(math.sqrt)

    // Since we implicitly do the feature scaling when we compute the cost function
    // to improve the convergence, the effective regParam will be changed.
    val effectiveRegParam = $(regParam) / yStd
    val effectiveL1RegParam = $(elasticNetParam) * effectiveRegParam
    val effectiveL2RegParam = (1.0 - $(elasticNetParam)) * effectiveRegParam

    val costFun = new LassoCostFun(instances, yStd, yMean,
      featuresStd, featuresMean, effectiveL2RegParam,effectiveL1RegParam)

    val optimizer = if ($(elasticNetParam) == 0.0 || effectiveRegParam == 0.0) {
      new BreezeLBFGS[BDV[Double]]($(maxIter), 10, $(tol))
    } else {
      new BreezeOWLQN[Int, BDV[Double]]($(maxIter), 10, effectiveL1RegParam, $(tol))
    }

    val initialWeights = Vectors.zeros(numFeatures)
    val states =
      optimizer.iterations(new CachedDiffFunction(costFun), initialWeights.toBreeze.toDenseVector)

    var state = states.next()
    val lossHistory = mutable.ArrayBuilder.make[Double]

    while (states.hasNext) {
      lossHistory += state.value
      state = states.next()
    }
    lossHistory += state.value

    // The weights are trained in the scaled space; we're converting them back to
    // the original space.

    val weights = {
      val rawWeights = state.x.toArray.clone()
      var i = 0
      val len = rawWeights.length
      while (i < len) {
        rawWeights(i) *= { if (featuresStd(i) != 0.0) yStd / featuresStd(i) else 0.0 }
        i += 1
      }
      Vectors.dense(rawWeights)
    }

//    val weights = Vectors.dense(state.x.toArray.clone())

    // The intercept in R's GLMNET is computed using closed form after the coefficients are
    // converged. See the following discussion for detail.
    // http://stats.stackexchange.com/questions/13617/how-is-the-intercept-computed-in-glmnet
    val intercept = yMean - dot(weights, Vectors.dense(featuresMean))
    if (handlePersistence) instances.unpersist()

    // TODO: Converts to sparse format based on the storage, but may base on the scoring speed.
    copyValues(new LinearRegressionModel(uid, weights.compressed, intercept))
  }

  override def copy(extra: ParamMap): LinearRegression = defaultCopy(extra)
}


private class LoggingLeastSquaresAggregator(
                                      weights: Vector,
                                      labelStd: Double,
                                      labelMean: Double,
                                      featuresStd: Array[Double],
                                      featuresMean: Array[Double]) extends Serializable {

  private var totalCnt: Long = 0L
  private var lossSum = 0.0

  private val (effectiveWeightsArray: Array[Double], offset: Double, dim: Int) = {
    val weightsArray = weights.toArray.clone()
    var sum = 0.0
    var i = 0
    val len = weightsArray.length
    while (i < len) {
      if (featuresStd(i) != 0.0) {
        weightsArray(i) /=  featuresStd(i)
        sum += weightsArray(i) * featuresMean(i)
      } else {
        weightsArray(i) = 0.0
      }
      i += 1
    }
    (weightsArray, -sum + labelMean / labelStd, weightsArray.length)
  }

//  private val (effectiveWeightsArray: Array[Double], offset: Double, dim: Int) = {
//    val weightsArray = weights.toArray.clone()
//    var sum = 0.0
//    var i = 0
//    val len = weightsArray.length
//    (weightsArray,0.0,weightsArray.length)
//  }

  private val effectiveWeightsVector = Vectors.dense(effectiveWeightsArray)

  private val gradientSumArray = Array.ofDim[Double](dim)

  /**
   * Add a new training data to this LeastSquaresAggregator, and update the loss and gradient
   * of the objective function.
   *
   * @param label The label for this data point.
   * @param data The features for one data point in dense/sparse vector format to be added
   *             into this aggregator.
   * @return This LeastSquaresAggregator object.
   */
  def add(label: Double, data: Vector): this.type = {
    require(dim == data.size, s"Dimensions mismatch when adding new sample." +
      s" Expecting $dim but got ${data.size}.")

    val diff = dot(data, effectiveWeightsVector) - label / labelStd + offset

    if (diff != 0) {
      val localGradientSumArray = gradientSumArray
      data.foreachActive { (index, value) =>
        if (featuresStd(index) != 0.0 && value != 0.0) {
          localGradientSumArray(index) += diff * value / featuresStd(index)
        }
      }
      lossSum += diff * diff / 2.0
    }

    totalCnt += 1
    this
  }

  /**
   * Merge another LeastSquaresAggregator, and update the loss and gradient
   * of the objective function.
   * (Note that it's in place merging; as a result, `this` object will be modified.)
   *
   * @param other The other LeastSquaresAggregator to be merged.
   * @return This LeastSquaresAggregator object.
   */
  def merge(other: LoggingLeastSquaresAggregator): this.type = {
    require(dim == other.dim, s"Dimensions mismatch when merging with another " +
      s"LeastSquaresAggregator. Expecting $dim but got ${other.dim}.")

    if (other.totalCnt != 0) {
      totalCnt += other.totalCnt
      lossSum += other.lossSum

      var i = 0
      val localThisGradientSumArray = this.gradientSumArray
      val localOtherGradientSumArray = other.gradientSumArray
      while (i < dim) {
        localThisGradientSumArray(i) += localOtherGradientSumArray(i)
        i += 1
      }
    }
    this
  }

  def count: Long = totalCnt

  def loss: Double = lossSum / totalCnt

  def gradient: Vector = {
    val result = Vectors.dense(gradientSumArray.clone())
    scal(1.0 / totalCnt, result)
    result
  }
}



/**
 * LeastSquaresCostFun implements Breeze's DiffFunction[T] for Least Squares cost.
 * It returns the loss and gradient with L2 regularization at a particular point (weights).
 * It's used in Breeze's convex optimization routines.
 */
private class LassoCostFun(
                                   data: RDD[(Double, Vector)],
                                   labelStd: Double,
                                   labelMean: Double,
                                   featuresStd: Array[Double],
                                   featuresMean: Array[Double],
                                   effectiveL2regParam: Double,
                                   effectiveL1regParam: Double
                                   ) extends DiffFunction[BDV[Double]] {

  override def calculate(weights: BDV[Double]): (Double, BDV[Double]) = {
    val w = Vectors.fromBreeze(weights)

    val leastSquaresAggregator = data.treeAggregate(new LoggingLeastSquaresAggregator(w, labelStd,
      labelMean, featuresStd, featuresMean))(
        seqOp = (c, v) => (c, v) match {
          case (aggregator, (label, features)) => aggregator.add(label, features)
        },
        combOp = (c1, c2) => (c1, c2) match {
          case (aggregator1, aggregator2) => aggregator1.merge(aggregator2)
        })

    // regVal is the sum of weight squares for L2 regularization
    val norm = brzNorm(weights, 2.0)
    val regVal = 0.5 * effectiveL2regParam * norm * norm



    //record objective Value just for test
    val primalObjective  = CostFun.calculate (w, data, new LeastSquaresGradient, new L1Updater(), 0.001, data.count () )
    val pw = new PrintWriter(new BufferedWriter(new FileWriter("output/primal_" +"lasso"+".txt", true)))
    pw.println(primalObjective)
    pw.flush()
    pw.close

    //record weights

    //

    //record time

    //

    //compare the result

    val loss = leastSquaresAggregator.loss + regVal
    val gradient = leastSquaresAggregator.gradient
    axpy(effectiveL2regParam, w, gradient)



    (loss, gradient.toBreeze.asInstanceOf[BDV[Double]])
  }
}
