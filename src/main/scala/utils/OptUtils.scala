package MLbenchmark.utils

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD
import breeze.linalg.{NumericOps, DenseVector, SparseVector, Vector}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.linalg._


object OptUtils {

  // load data stored in LIBSVM format
  def loadLIBSVMData(sc: SparkContext, filename: String, numSplits: Int, numFeats: Int): RDD[LabeledPoint] = {

    // read in text file
    val data = sc.textFile(filename, numSplits).coalesce(numSplits)  // note: coalesce can result in data being sent over the network. avoid this for large datasets
    val numEx = data.count()

    // find number of elements per partition
    val numParts = data.partitions.size
    val sizes = data.mapPartitionsWithSplit{ case(i, lines) =>
      Iterator(i -> lines.length)
    }.collect().sortBy(_._1)
    val offsets = sizes.map(x => x._2).scanLeft(0)(_+_).toArray

    // parse input
    data.mapPartitionsWithSplit { case(partition, lines) =>
      lines.zipWithIndex.flatMap{ case(line, idx) =>

        // calculate index for line
        val index = offsets(partition) + idx

        if(index < numEx){

          // parse label
          val parts = line.trim().split(' ')
          var label = -1
          if (parts(0).contains("+") || parts(0).toInt == 1)
            label = 1

          // parse features
          val featureArray = parts.slice(1, parts.length)
            .map(_.split(':') 
            match { case Array(i,j) => (i.toInt-1, j.toDouble)}).toArray
          val features = Vectors.sparse(numFeats, featureArray.map(x=>x._1), featureArray.map(x=>x._2))
          // create classification point
          Iterator(LabeledPoint(label,features))
        }
        else{
          Iterator()
        }
      }
    }
  }
  
}
