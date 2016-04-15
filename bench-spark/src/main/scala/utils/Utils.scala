package utils

import breeze.linalg.{DenseVector, SparseVector}
import l1distopt.utils.OptUtils
import org.apache.spark.SparkContext
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD

import scala.xml.XML

/**
  * Created by amirreza on 12/04/16.
  */
object Utils {
  def loadLibSVMForBinaryClassification(dataset: String, numPartitions: Int = 4, sc: SparkContext): RDD[LabeledPoint] = {
    val xml = XML.loadFile("configs.xml")
    val projectPath = (xml \\ "config" \\ "projectpath") text
    //Load data
    val data: RDD[LabeledPoint] = MLUtils.loadLibSVMFile(sc,
      projectPath + "datasets/" + dataset)


    //Take only two class with labels -1 and +1 for binary classification
    val points = data.filter(p => p.label == 3.0 || p.label == 2.0).
      map(p => if (p.label == 2.0) LabeledPoint(-1.0, p.features)
      else LabeledPoint(+1.0, p.features)).repartition(numPartitions)
    return points
  }

  def loadLibSVMForRegressionProxCocoaFormat(dataset: String, numPartitions: Int = 4, sc: SparkContext):
  RDD[l1distopt.utils.LabeledPoint] = {
    val xml = XML.loadFile("configs.xml")
    val projectPath = (xml \\ "config" \\ "projectpath") text
    //Load data
    val regData = OptUtils.loadLIBSVMData(sc, projectPath + "datasets/" + dataset, numPartitions, 10)

    return regData
  }

  def loadLibSVMForRegressionProxCocoaTrainFormat(dataset: String, numPartitions: Int = 4, sc: SparkContext):
  (RDD[(Int, SparseVector[Double])], DenseVector[Double]) = {
    val xml = XML.loadFile("configs.xml")
    val projectPath = (xml \\ "config" \\ "projectpath") text
    //Load data
    val regData = OptUtils.loadLIBSVMDataColumn(sc, projectPath + "datasets/" + dataset, numPartitions, 10)

    return regData
  }
}
