package utils

import Functions.{ProxCocoaDataMatrix, SGDDataMatrix}
import breeze.linalg.{DenseVector, SparseVector}
import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.Vectors
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

  def loadLibSVMForRegressionProxCocoa(dataset: String, numPartitions: Int = 4, numFeats: Int, sc: SparkContext):
  (ProxCocoaDataMatrix, ProxCocoaDataMatrix,
    SGDDataMatrix, SGDDataMatrix,
    RDD[l1distopt.utils.LabeledPoint], RDD[l1distopt.utils.LabeledPoint]) = {

    val xml = XML.loadFile("configs.xml")
    val projectPath = (xml \\ "config" \\ "projectpath") text
    val filename: String  = projectPath + "datasets/" + dataset
    //Load data
    val data = sc.textFile(filename,numPartitions).coalesce(numPartitions)
    val Array(train, test) = data.randomSplit(Array(0.8, 0.2), seed = 13)
    val trainColumn = rowToColumnProxCocoa(train)
    val testColumn = rowToColumnProxCocoa(test)
    val trainProx = toRowProxCocoa(train, numFeats)
    val testProx = toRowProxCocoa(test, numFeats)
    val trainNormal = trainProx.map(p => LabeledPoint(p.label, Vectors.dense(p.features.toArray)))
    val testNormal = testProx.map(p => LabeledPoint(p.label, Vectors.dense(p.features.toArray)))

    return (trainColumn, testColumn, trainNormal, testNormal, trainProx, testProx)
  }

  def toRowProxCocoa(data: RDD[String],
                     numFeats: Int): RDD[l1distopt.utils.LabeledPoint] = {
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
          var label = parts(0).toDouble

          // parse features
          val featureArray = parts.slice(1, parts.length)
            .map(_.split(':')
            match { case Array(i, j) => (i.toInt - 1, j.toDouble)
            }).toArray
          var features = new SparseVector[Double](featureArray.map(x => x._1),
            featureArray.map(x => x._2), numFeats)

          // create classification point
          Iterator(l1distopt.utils.LabeledPoint(label, features))
        }
        else {
          Iterator()
        }
      }
    }
  }

  def rowToColumnProxCocoa(data: RDD[String]): ProxCocoaDataMatrix = {

    val numEx = data.count().toInt

    // find number of elements per partition
    val numParts = data.partitions.size
    val sizes = data.mapPartitionsWithSplit{ case(i,lines) =>
      Iterator(i -> lines.length)
    }.collect().sortBy(_._1)
    val offsets = sizes.map(x => x._2).scanLeft(0)(_+_).toArray

    // parse input
    val parsedData = data.mapPartitionsWithSplit { case(partition, lines) =>
      lines.zipWithIndex.flatMap{ case(line, idx) =>

        // calculate index for line
        val index = offsets(partition) + idx

        if(index < numEx) {

          // parse label
          val parts = line.trim().split(' ')
          var label = parts(0).toDouble

          // parse features
          val featureArray = parts.slice(1,parts.length)
            .map(_.split(':')
            match { case Array(i,j) => (i.toInt-1, (index, j.toDouble))}).toArray
          Iterator((label, featureArray))
        }
        else {
          Iterator()
        }
      }
    }

    // collect all of the labels
    val y = new DenseVector[Double](parsedData.map(x => x._1).collect())

    // arrange RDD by feature
    val feats = parsedData.flatMap(x => x._2.iterator)
      .groupByKey().map(x => (x._1, x._2.toArray)).map(x => (x._1, new SparseVector[Double](x._2.map(y => y._1), x._2.map(y => y._2), numEx)))

    // return data and labels
    println("successfully loaded training data")
    return (feats,y)
  }
}
