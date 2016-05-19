package utils

import java.io.File

import Functions.{CocoaLabeledPoint, ProxCocoaDataMatrix, ProxCocoaLabeledPoint}
import breeze.linalg.{DenseVector, SparseVector}
import l1distopt.utils.{DebugParams, Params}
import optimizers.{CocoaParameters, LBFGSParameters, ProxCocoaParameters, SGDParameters}
import org.apache.spark.SparkContext
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD

import scala.xml.XML

/**
  * Created by amirreza on 12/04/16.
  */
object Utils {
  /**
    * Loads all different partitions already saved using MLUtils.saveAsLibSVMFile(...) contained in directory given by
    * argument. Name of all partitions starts with part*.
    *
    * @param dir directory in which partitions are saved.
    * @param sc SparkContext
    * @return RDD[LabeledPoint] containing all partitions
    */
  def loadLibSVMFromDir(dir: String, sc: SparkContext): RDD[LabeledPoint] = {
    val files = (new File(dir)).listFiles.filter(_.getName.startsWith("part")).map(_.getName)
    return files.map(part => MLUtils.loadLibSVMFile(sc, dir + part)).reduceLeft(_.union(_))
  }

  /**
    * Loads LibSvm dataset and split it to train 4/5th and test with 1/5th
    *
    * @param dataset absolute address of the LibSvm dataset
    * @param numPartitions
    * @param sc SparkContext
    * @return train and test data split from original data
    */
  def loadAbsolutLibSVMRegression(dataset: String, numPartitions: Int = 4, sc: SparkContext):
  (RDD[LabeledPoint], RDD[LabeledPoint]) = {
    //Load data
    val data: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, dataset).repartition(numPartitions)
    val Array(train, test) = data.randomSplit(Array(0.8, 0.2), seed = 13)
    return (train, test)
  }
  def loadAbsolutLibSVMBinaryClassification(dataset: String, numPartitions: Int = 4, sc: SparkContext):
  (RDD[LabeledPoint], RDD[LabeledPoint]) = {
    //Load data
    val data: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, dataset).repartition(numPartitions)
    val Array(train, test) = data.randomSplit(Array(0.8, 0.2), seed = 13)
    return (train, test)
  }

  def loadAbsolutLibSVMForBinaryClassification(dataset: String, a: Int, b: Int, numPartitions: Int = 4, sc: SparkContext):
  (RDD[LabeledPoint], RDD[LabeledPoint]) = {
    //Load data
    val data: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc, dataset).repartition(numPartitions)
    //Take only two class with labels -1 and +1 for binary classification
    val points = data.filter(p => p.label == a || p.label == b).
      map(p => if (p.label == b) LabeledPoint(-1.0, p.features)
      else LabeledPoint(+1.0, p.features)).repartition(numPartitions)

    val Array(train, test) = points.randomSplit(Array(0.8, 0.2), seed = 13)
    return (train, test)
  }
  def toMllibClassificationlabeling(data: RDD[LabeledPoint]): RDD[LabeledPoint] = {
    val points = data.map(p => if(p.label == -1) LabeledPoint(0, p.features)
    else LabeledPoint(p.label, p.features))
    return points
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

  def defaultDebugCocoa(train: RDD[LabeledPoint], test: RDD[LabeledPoint]): distopt.utils.DebugParams = {
    val debugIter = 10 // set to -1 to turn off debugging output
    val seed = 13 // set seed for debug purposes
    var chkptIter = 100
    val debug = distopt.utils.DebugParams(Utils.toCocoaFormat(test), debugIter, seed, chkptIter)
    return debug
  }

  def defaultDebugProxCocoa(train: RDD[LabeledPoint], test: RDD[LabeledPoint]): l1distopt.utils.DebugParams = {
    val seed = 13
    val debugIter = 10
    val debug = DebugParams(Utils.toProxCocoaFormat(test), debugIter, seed)
    return debug
  }

  def readGDParameters(workingDir: String): SGDParameters ={
    val params = new SGDParameters(miniBatchFraction = 1.0)
    if(new java.io.File(workingDir + "parameters.xml").exists()) {
      val xml = XML.loadFile(workingDir + "parameters.xml")
      params.iterations = (xml \\ "Parameters" \\ "GDParameters" \\ "iterations").text.toInt
      params.miniBatchFraction = (xml \\ "Parameters" \\ "GDParameters" \\ "miniBatchFraction").text.toDouble
      params.stepSize = (xml \\ "Parameters" \\ "GDParameters" \\ "stepSize").text.toDouble
      params.seed = (xml \\ "Parameters" \\ "GDParameters" \\ "seed").text.toInt
    }
    require(params.miniBatchFraction == 1.0, s"Use optimizers.SGD for miniBatchFraction less than 1.0")
    params
  }
  def readSGDParameters(workingDir: String): SGDParameters = {
    val params = new SGDParameters(miniBatchFraction = Functions.DEFAULT_BATCH_FRACTION)
    if(new java.io.File(workingDir + "parameters.xml").exists()) {
      val xml = XML.loadFile(workingDir + "parameters.xml")
      params.iterations = (xml \\ "Parameters" \\ "SGDParameters" \\ "iterations").text.toInt
      params.miniBatchFraction = (xml \\ "Parameters"  \\ "SGDParameters" \\ "miniBatchFraction").text.toDouble
      params.stepSize = (xml \\ "Parameters" \\ "SGDParameters" \\ "stepSize").text.toDouble
      params.seed = (xml \\ "Parameters" \\ "SGDParameters" \\ "seed").text.toInt
    }
    require(params.miniBatchFraction < 1.0, "miniBatchFraction must be less than 1. Use GD otherwise.")
    params
  }
  def readLBFGSParameters(workingDir: String): LBFGSParameters = {
    val params = new LBFGSParameters()
    if(new java.io.File(workingDir + "parameters.xml").exists()) {
      val xml = XML.loadFile(workingDir + "parameters.xml")

      params.iterations = (xml \\ "Parameters" \\ "LBFGSParameters" \\ "iterations").text.toInt
      params.numCorrections = (xml \\ "Parameters" \\ "LBFGSParameters" \\ "numCorrections").text.toInt
      params.convergenceTol = (xml \\ "Parameters" \\ "LBFGSParameters" \\ "convergenceTol").text.toDouble
      params.seed = (xml \\ "Parameters" \\ "LBFGSParameters" \\ "seed").text.toInt
    }
    params
  }
  def readCocoaParameters(workingDir: String, train: RDD[LabeledPoint], test: RDD[LabeledPoint]): CocoaParameters = {
    val params = new CocoaParameters(train, test)
    if(new java.io.File(workingDir + "parameters.xml").exists()) {
      val xml = XML.loadFile(workingDir + "parameters.xml")
      params.numRounds = (xml \\ "Parameters" \\ "CocoaParameters" \\ "numRounds").text.toInt
      params.localIterFrac = (xml \\ "Parameters" \\ "CocoaParameters" \\ "localIterFrac").text.toDouble
      params.lambda = (xml \\ "Parameters" \\ "CocoaParameters" \\ "lambda").text.toDouble
      params.beta = (xml \\ "Parameters" \\ "CocoaParameters" \\ "beta").text.toDouble
      params.gamma = (xml \\ "Parameters" \\ "CocoaParameters" \\ "gamma").text.toDouble
    }
    params
  }

  def readElasticProxCocoaParameters(workingDir: String, train: RDD[LabeledPoint], test: RDD[LabeledPoint]): ProxCocoaParameters = {
    val params = new ProxCocoaParameters(train, test)
    if(new java.io.File(workingDir + "parameters.xml").exists()) {
      val xml = XML.loadFile(workingDir + "parameters.xml")
      params.iterations = (xml \\ "Parameters" \\ "ElasticProxCocoaParameters" \\ "iterations").text.toInt
      params.localIterFrac = (xml \\ "Parameters" \\ "ElasticProxCocoaParameters" \\ "localIterFrac").text.toDouble
      params.lambda = (xml \\ "Parameters" \\ "ElasticProxCocoaParameters" \\ "lambda").text.toDouble
      params.eta = (xml \\ "Parameters" \\ "ElasticProxCocoaParameters" \\ "eta").text.toDouble
    }
    params
  }
  def readL1ProxCocoaParameters(workingDir: String, train: RDD[LabeledPoint], test: RDD[LabeledPoint]): ProxCocoaParameters = {
    val params = new ProxCocoaParameters(train, test)
    if(new java.io.File(workingDir + "parameters.xml").exists()) {
      val xml = XML.loadFile(workingDir + "parameters.xml")
      params.iterations = (xml \\ "Parameters" \\ "L1ProxCocoaParameters" \\ "iterations").text.toInt
      params.localIterFrac = (xml \\ "Parameters" \\ "L1ProxCocoaParameters" \\ "localIterFrac").text.toDouble
      params.lambda = (xml \\ "Parameters" \\ "L1ProxCocoaParameters" \\ "lambda").text.toDouble
      params.eta = (xml \\ "Parameters" \\ "L1ProxCocoaParameters" \\ "eta").text.toDouble
    }
    require(params.eta == 1.0, "eta must be 1 for L1-regularization")
    params
  }
}
