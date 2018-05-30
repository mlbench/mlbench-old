import org.apache.log4j.{Level, Logger}
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.{SparkConf, SparkContext}
import utils.Utils

import scala.util.Try

import scalax.file.Path


class CLIParserDataGen(arguments: Seq[String]) extends org.rogach.scallop.ScallopConf(arguments) {
  val numPoints = opt[Int](required = true, short = 'n', descr = "Number of data points to generate")
  val numFeatures = opt[Int](required = true, short = 'm', descr = "Number of features to generate")
  val partitions = opt[Int](required = false, default = Some(4), short = 'p', validate = (0 <),
    descr = "Number of spark partitions to be used. Optional.")
  val dir = opt[String](required = true, default = Some("../dataset"), short = 'd', descr = "working directory where dataset is stored. Default is \"../results\". ")
  val datasetType = opt[String](required = false, default = Some("Regression"), descr = "Type of dataset. Can be \"Regression\" for the moment.")
  verify()
}


object GenerateData {
  def main(args: Array[String]) {
    //Spark conf
    val conf = new SparkConf().setAppName("Distributed Machine Learning").setMaster("local[*]")
    val sc = new SparkContext(conf)
    val sqlContext = new org.apache.spark.sql.SQLContext(sc)

    //Parser arguments
    val parser = new CLIParserDataGen(args)
    val numPoints = parser.numPoints()
    val numFeatures = parser.numFeatures()
    val numPartitions = parser.partitions()
    val workingDir = parser.dir()
    val datasetType = parser.datasetType()

    if ( datasetType == "Regression" ) {
      val data = Utils.generateLabeledPoints(sc, numPoints, numFeatures, 1, 1.0, numPartitions, System.nanoTime())
      MLUtils.saveAsLibSVMFile(data, workingDir)
    } else {
      print("Error: dataset generation of type \"" + datasetType + "\" not supported.")
      System.exit(1)
    }
  }
}
