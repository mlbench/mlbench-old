package utils

import java.io.File

import Functions.{CocoaLabeledPoint, ProxCocoaDataMatrix, ProxCocoaLabeledPoint}
import breeze.linalg.{DenseVector, SparseVector}
import l1distopt.utils.{DebugParams, Params}
import org.apache.spark.SparkContext
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD

import scala.xml.XML

/**
  * Created by amirreza on 12/04/16.
  */
object Utils {
  def loadLibSVMFromDir(dir: String, sc: SparkContext): RDD[LabeledPoint] = {
    val files = (new File(dir)).listFiles.filter(_.getName.startsWith("part")).map(_.getName)
    return files.map(part => MLUtils.loadLibSVMFile(sc, dir + part).repartition(1)).reduceLeft(_.union(_))
  }

  def loadRawDataset(dataset: String, sc: SparkContext): RDD[LabeledPoint] = {
    // get projectPath from config
    val xml = XML.loadFile("configs.xml")
    val projectPath = (xml \\ "config" \\ "projectpath") text
    val datasetPath = projectPath + (if (projectPath.endsWith("/")) "" else "/") + "datasets/" + dataset
    //Load data
    val data: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, datasetPath)
    return data
  }

  def loadLibSVMForBinaryClassification(dataset: String, numPartitions: Int = 4, sc: SparkContext):
  (RDD[LabeledPoint], RDD[LabeledPoint]) = {
    val data = loadRawDataset(dataset, sc)
    //Take only two class with labels -1 and +1 for binary classification
    val points = data.filter(p => p.label == 3.0 || p.label == 2.0).
      map(p => if (p.label == 2.0) LabeledPoint(-1.0, p.features)
      else LabeledPoint(+1.0, p.features)).repartition(numPartitions)

    val Array(train, test) = points.randomSplit(Array(0.8, 0.2), seed = 13)
    return (train, test)
  }

  def loadLibSVMBinaryClassification(dataset: String, numPartitions: Int = 4, sc: SparkContext):
  (RDD[LabeledPoint], RDD[LabeledPoint]) = {
    val data = loadRawDataset(dataset, sc)
    val Array(train, test) = data.randomSplit(Array(0.8, 0.2), seed = 13)
    return (train, test)
  }

  def loadLibSVMRegression(dataset: String, numPartitions: Int = 4, sc: SparkContext):
  (RDD[LabeledPoint], RDD[LabeledPoint]) = {
    val xml = XML.loadFile("configs.xml")
    val projectPath = (xml \\ "config" \\ "projectpath") text
    //Load data
    val data: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc,
      projectPath + "datasets/" + dataset).repartition(numPartitions)
    val Array(train, test) = data.randomSplit(Array(0.8, 0.2), seed = 13)
    return (train, test)
  }

  def loadAbsolutLibSVMRegression(dataset: String, numPartitions: Int = 4, sc: SparkContext):
  (RDD[LabeledPoint], RDD[LabeledPoint]) = {
    //Load data
    val data: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, dataset).repartition(numPartitions)
    val Array(train, test) = data.randomSplit(Array(0.8, 0.2), seed = 13)
    return (train, test)
  }

  def loadAbsolutLibSVMForBinaryClassification(dataset: String, numPartitions: Int = 4, sc: SparkContext):
  (RDD[LabeledPoint], RDD[LabeledPoint]) = {
    //Load data
    val data: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, dataset).repartition(numPartitions)
    //Take only two class with labels -1 and +1 for binary classification
    val points = data.filter(p => p.label == 3.0 || p.label == 2.0).
      map(p => if (p.label == 2.0) LabeledPoint(-1.0, p.features)
      else LabeledPoint(+1.0, p.features)).repartition(numPartitions)

    val Array(train, test) = points.randomSplit(Array(0.8, 0.2), seed = 13)
    return (train, test)
  }

  def loadAbsolutLibSVMBinaryClassification(dataset: String, numPartitions: Int = 4, sc: SparkContext):
  (RDD[LabeledPoint], RDD[LabeledPoint]) = {
    //Load data
    val data: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, dataset).repartition(numPartitions)
    val Array(train, test) = data.randomSplit(Array(0.8, 0.2), seed = 13)
    return (train, test)
  }

  def toProxCocoaTranspose(data: RDD[LabeledPoint]): ProxCocoaDataMatrix = {
    val numEx = data.count().toInt
    val myData: RDD[(Double, Array[(Int, (Int, Double))])] = data.zipWithIndex().
      map(p => (p._2, p._1.label, p._1.features.toArray.zipWithIndex)).
      map(x => (x._2, x._3.map(p => (p._2, (x._1.toInt, p._1)))))

    val y: DenseVector[Double] = new DenseVector[Double](myData.map(x => x._1).collect())
    // arrange RDD by feature
    val feats: RDD[(Int, SparseVector[Double])] = myData.flatMap(x => x._2.iterator)
      .groupByKey().map(x => (x._1, x._2.toArray)).map(x => (x._1, new SparseVector[Double](x._2.map(y => y._1), x._2.map(y => y._2), numEx)))
    return (feats, y)
  }

  def toProxCocoaFormat(data: RDD[LabeledPoint]): ProxCocoaLabeledPoint = {
    data.map(p => l1distopt.utils.LabeledPoint(p.label, SparseVector(p.features.toArray.map(x => x.toDouble))))
  }

  def toCocoaFormat(data: RDD[LabeledPoint]): CocoaLabeledPoint = {
    data.map(p => distopt.utils.LabeledPoint(p.label, SparseVector(p.features.toArray.map(x => x.toDouble))))
  }


  def defaultL1ProxParams(train: RDD[LabeledPoint], test: RDD[LabeledPoint]): (
    l1distopt.utils.Params, l1distopt.utils.DebugParams) = {
    val seed = 13
    //Regularization parameters
    val lambda = 0.1
    val eta = 1.0
    //optimization parameters
    val iterations = 100
    val localIterFrac = 0.9
    val debugIter = 10
    val force_cache = train.count().toInt
    val n = train.count().toInt
    var localIters = (localIterFrac * train.first().features.size / train.partitions.size).toInt
    localIters = Math.max(localIters, 1)
    val alphaInit = SparseVector.zeros[Double](10)
    val proxParams = Params(alphaInit, n, iterations, localIters, lambda, eta)
    val debug = DebugParams(Utils.toProxCocoaFormat(test), debugIter, seed)
    return (proxParams, debug)
  }


  def defaultCocoa(train: RDD[LabeledPoint], test: RDD[LabeledPoint]):
  (distopt.utils.Params, distopt.utils.DebugParams) = {
    val lambda = 0.01
    val numRounds = 200 // number of outer iterations, called T in the paper
    val localIterFrac = 1.0 // fraction of local points to be processed per round, H = localIterFrac * n
    val beta = 1.0 // scaling parameter when combining the updates of the workers (1=averaging for CoCoA)
    val gamma = 1.0 // aggregation parameter for CoCoA+ (1=adding, 1/K=averaging)
    val debugIter = 10 // set to -1 to turn off debugging output
    val seed = 13 // set seed for debug purposes
    val n = train.count().toInt
    var localIters = (localIterFrac * n / train.partitions.size).toInt
    localIters = Math.max(localIters, 1)
    var chkptIter = 100
    val wInit = DenseVector.zeros[Double](train.first().features.size)
    // set to solve hingeloss SVM
    val loss = distopt.utils.OptUtils.hingeLoss _
    val params = distopt.utils.Params(loss, n, wInit, numRounds, localIters, lambda, beta, gamma)
    val debug = distopt.utils.DebugParams(Utils.toCocoaFormat(test), debugIter, seed, chkptIter)
    return (params, debug)
  }

  def defaultElasticProxParams(train: RDD[LabeledPoint], test: RDD[LabeledPoint]): (
    l1distopt.utils.Params, l1distopt.utils.DebugParams) = {
    val seed = 13
    //Regularization parameters
    val lambda = 0.1
    val eta = 0.5
    //optimization parameters
    val iterations = 100
    val localIterFrac = 0.9
    val debugIter = 10
    val force_cache = train.count().toInt
    val n = train.count().toInt
    var localIters = (localIterFrac * train.first().features.size / train.partitions.size).toInt
    localIters = Math.max(localIters, 1)
    val alphaInit = SparseVector.zeros[Double](10)
    val proxParams = Params(alphaInit, n, iterations, localIters, lambda, eta)
    val debug = DebugParams(Utils.toProxCocoaFormat(test), debugIter, seed)
    return (proxParams, debug)
  }
}
