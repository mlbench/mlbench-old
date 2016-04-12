package utils

import org.apache.spark.SparkContext
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD

import scala.xml.XML

/**
  * Created by amirreza on 12/04/16.
  */
object Utils {
  def loadLibSVMForBinaryClassification(dataset:String, numPartitions: Int = 4, sc:SparkContext): RDD[LabeledPoint] = {
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

}
