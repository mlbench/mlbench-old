package utils

import java.io.{File, FileInputStream}

//import Functions.{CocoaLabeledPoint, ProxCocoaDataMatrix, ProxCocoaLabeledPoint}
import breeze.linalg.{DenseVector, SparseVector}
//import l1distopt.utils.DebugParams
import optimizers.{SGDParameters}
import org.apache.spark.SparkContext
import org.apache.spark.mllib.random._
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.util.MLUtils
import org.apache.spark.rdd.RDD
import org.yaml.snakeyaml.Yaml

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
  /*def toProxCocoaTranspose(data: RDD[LabeledPoint]): ProxCocoaDataMatrix = {
    val numEx = data.count().toInt
    val myData: RDD[(Double, Array[(Int, (Int, Double))])] = data.zipWithIndex().
      map(p => (p._2, p._1.label, p._1.features.toArray.zipWithIndex)).
      map(x => (x._2, x._3.map(p => (p._2, (x._1.toInt, p._1)))))

    val y: DenseVector[Double] = new DenseVector[Double](myData.map(x => x._1).collect())
    // arrange RDD by feature
    val feats: RDD[(Int, SparseVector[Double])] = myData.flatMap(x => x._2.iterator)
      .groupByKey().map(x => (x._1, x._2.toArray)).map(x => (x._1, new SparseVector[Double](x._2.map(y => y._1), x._2.map(y => y._2), numEx)))
    return (feats, y)
  }*/

  /*def toProxCocoaFormat(data: RDD[LabeledPoint]): ProxCocoaLabeledPoint = {
    data.map(p => l1distopt.utils.LabeledPoint(p.label, SparseVector(p.features.toArray.map(x => x.toDouble))))
  }*/

  /*def toCocoaFormat(data: RDD[LabeledPoint]): CocoaLabeledPoint = {
    data.map(p => distopt.utils.LabeledPoint(p.label, SparseVector(p.features.toArray.map(x => x.toDouble))))
  }*/

  /*def defaultDebugCocoa(train: RDD[LabeledPoint], test: RDD[LabeledPoint]): distopt.utils.DebugParams = {
    val debugIter = 10 // set to -1 to turn off debugging output
    val seed = 13 // set seed for debug purposes
    var chkptIter = 100
    val debug = distopt.utils.DebugParams(Utils.toCocoaFormat(test), debugIter, seed, chkptIter)
    return debug
  }*/

  /*def defaultDebugProxCocoa(train: RDD[LabeledPoint], test: RDD[LabeledPoint]): l1distopt.utils.DebugParams = {
    val seed = 13
    val debugIter = 10
    val debug = DebugParams(Utils.toProxCocoaFormat(test), debugIter, seed)
    return debug
  }*/

  def readGDParameters(workingDir: String): SGDParameters ={
    val params = new SGDParameters(miniBatchFraction = 1.0)
    if(new java.io.File(workingDir + "parameters.yaml").exists()) {
      val file = new FileInputStream(new File(workingDir + "parameters.yaml"));
      val yaml = new Yaml()
      val data = yaml.load(file).asInstanceOf[java.util.Map[String, java.util.Map[String, java.util.Map[String, Object]]]]
      if (data != null) {
        val optParams = data.get("OptParameters")
        if (optParams != null) {
          val sgdParams = optParams.get("GDParameters")
          if (sgdParams != null) {
            val iterations = sgdParams.get("iterations").asInstanceOf[Int]
            if (iterations != 0) {
              params.iterations = iterations
            }
            val miniBatchFraction = sgdParams.get("miniBatchFraction").asInstanceOf[Double]
            if (miniBatchFraction != 0) {
              params.miniBatchFraction = miniBatchFraction
            }
            val stepSize = sgdParams.get("stepSize").asInstanceOf[Double]
            if (stepSize != 0) {
              params.stepSize = stepSize
            }
            val seed = sgdParams.get("seed").asInstanceOf[Int]
            if (seed != 0) {
              params.seed = seed
            }
          }
        }
      }
    }
    require(params.miniBatchFraction == 1.0, s"Use optimizers.SGD for miniBatchFraction less than 1.0")
    params
  }
  def readSGDParameters(workingDir: String): SGDParameters = {
    val params = new SGDParameters(miniBatchFraction = Functions.DEFAULT_BATCH_FRACTION)
    if(new java.io.File(workingDir + "parameters.yaml").exists()) {
      val file = new FileInputStream(new File(workingDir + "parameters.yaml"));
      val yaml = new Yaml()
      val data = yaml.load(file).asInstanceOf[java.util.Map[String, java.util.Map[String, java.util.Map[String, Object]]]]
      if(data != null){
      val optParams = data.get("OptParameters")
      if(optParams != null) {
        val sgdParams = optParams.get("SGDParameters")
        if (sgdParams != null) {
          val iterations = sgdParams.get("iterations").asInstanceOf[Int]
          if (iterations != 0) {
            params.iterations = iterations
          }
          val miniBatchFraction = sgdParams.get("miniBatchFraction").asInstanceOf[Double]
          if (miniBatchFraction != 0) {
            params.miniBatchFraction = miniBatchFraction
          }
          val stepSize = sgdParams.get("stepSize").asInstanceOf[Double]
          if (stepSize != 0) {
            params.stepSize = stepSize
          }
          val seed = sgdParams.get("seed").asInstanceOf[Int]
          if (seed != 0) {
            params.seed = seed
          }
        }
       }
      }
    }
    require(params.miniBatchFraction < 1.0, "miniBatchFraction must be less than 1. Use GD otherwise.")
    params
  }
  /*def readLBFGSParameters(workingDir: String): LBFGSParameters = {
    val params = new LBFGSParameters()
    if(new java.io.File(workingDir + "parameters.yaml").exists()) {
      val file = new FileInputStream(new File(workingDir + "parameters.yaml"));
      val yaml = new Yaml()
      val data = yaml.load(file).asInstanceOf[java.util.Map[String, java.util.Map[String, java.util.Map[String, Object]]]]
      if(data != null) {
        val optParams = data.get("OptParameters")
        if (optParams != null) {
          val lbfgsParams = optParams.get("LBFGSParameters")
          if (lbfgsParams != null) {
            val iterations = lbfgsParams.get("iterations").asInstanceOf[Int]
            if (iterations != 0) {
              params.iterations = iterations
            }
            val numCorrections = lbfgsParams.get("numCorrections").asInstanceOf[Int]
            if (numCorrections != 0) {
              params.numCorrections = numCorrections
            }
            val convergenceTol = lbfgsParams.get("convergenceTol").asInstanceOf[Double]
            if (convergenceTol != 0) {
              params.convergenceTol = convergenceTol
            }
            val seed = lbfgsParams.get("seed").asInstanceOf[Int]
            if (seed != 0) {
              params.seed = seed
            }
          }
        }
      }
    }
    params
  }*/
  /*def readCocoaParameters(workingDir: String, train: RDD[LabeledPoint], test: RDD[LabeledPoint]): CocoaParameters = {
    val params = new CocoaParameters(train, test)
    if(new java.io.File(workingDir + "parameters.yaml").exists()) {
      val file = new FileInputStream(new File(workingDir + "parameters.yaml"));
      val yaml = new Yaml()
      val data = yaml.load(file).asInstanceOf[java.util.Map[String, java.util.Map[String, java.util.Map[String, Object]]]]
      if(data != null) {
        val optParams = data.get("OptParameters")
        if (optParams != null) {
          val cocoaParams = optParams.get("CocoaParameters")
          if (cocoaParams != null) {
            val iterations = cocoaParams.get("numRounds").asInstanceOf[Int]
            if (iterations != 0) {
              params.numRounds = iterations
            }
            val localIterFrac = cocoaParams.get("localIterFrac").asInstanceOf[Double]
            if (localIterFrac != 0) {
              params.localIterFrac = localIterFrac
            }
            val lambda = cocoaParams.get("lambda").asInstanceOf[Double]
            if (lambda != 0) {
              params.lambda = lambda
            }
            val beta = cocoaParams.get("beta").asInstanceOf[Double]
            if (beta != 0) {
              params.beta = beta
            }
            val gamma = cocoaParams.get("gamma").asInstanceOf[Double]
            if (gamma != 0) {
              params.gamma = gamma
            }
          }
        }
      }
    }
    params
  }*/

  /*def readElasticProxCocoaParameters(workingDir: String, train: RDD[LabeledPoint], test: RDD[LabeledPoint]): ProxCocoaParameters = {
    val params = new ProxCocoaParameters(train, test)
    if(new java.io.File(workingDir + "parameters.yaml").exists()) {
      val file = new FileInputStream(new File(workingDir + "parameters.yaml"));
      val yaml = new Yaml()
      val data = yaml.load(file).asInstanceOf[java.util.Map[String, java.util.Map[String, java.util.Map[String, Object]]]]
      if(data != null) {
        val optParams = data.get("OptParameters")
        if (optParams != null) {
          val elasticParams = optParams.get("L1ProxCocoaParameters")
          if (elasticParams != null) {
            val iterations = elasticParams.get("iterations").asInstanceOf[Int]
            if (iterations != 0) {
              params.iterations = iterations
            }
            val localIterFrac = elasticParams.get("localIterFrac").asInstanceOf[Double]
            if (localIterFrac != 0) {
              params.localIterFrac = localIterFrac
            }
            val lambda = elasticParams.get("lambda").asInstanceOf[Double]
            if (lambda != 0) {
              params.lambda = lambda
            }
            val eta = elasticParams.get("eta").asInstanceOf[Double]
            if (eta != 0) {
              params.eta = eta
            }
          }
        }
      }
    }
    params
  }*/
  /*def readL1ProxCocoaParameters(workingDir: String, train: RDD[LabeledPoint], test: RDD[LabeledPoint]): ProxCocoaParameters = {
    val params = new ProxCocoaParameters(train, test)
    params.eta = 1.0
      if(new java.io.File(workingDir + "parameters.yaml").exists()) {
        val file = new FileInputStream(new File(workingDir + "parameters.yaml"));
        val yaml = new Yaml()
        val data = yaml.load(file).asInstanceOf[java.util.Map[String, java.util.Map[String, java.util.Map[String, Object]]]]
        if(data != null) {
          val optParams = data.get("OptParameters")
          if (optParams != null) {
            val proxParams = optParams.get("L1ProxCocoaParameters")
            if (proxParams != null) {
              val iterations = proxParams.get("iterations").asInstanceOf[Int]
              if (iterations != 0) {
                params.iterations = iterations
              }
              val localIterFrac = proxParams.get("localIterFrac").asInstanceOf[Double]
              if (localIterFrac != 0) {
                params.localIterFrac = localIterFrac
              }
              val lambda = proxParams.get("lambda").asInstanceOf[Double]
              if (lambda != 0) {
                params.lambda = lambda
              }
              val eta = proxParams.get("eta").asInstanceOf[Double]
              if (eta != 0) {
                params.eta = eta
              }
            }
          }
        }
    }
    require(params.eta == 1.0, "eta must be 1 for L1-regularization")
    params
  }*/

  def readRegParameters(workingDir: String): Double= {
    if(new java.io.File(workingDir + "parameters.yaml").exists()) {
      val file = new FileInputStream(new File(workingDir + "parameters.yaml"));
      val yaml = new Yaml()
      val data = yaml.load(file).asInstanceOf[java.util.Map[String, java.util.Map[String, java.util.Map[String, Object]]]]
      if(data != null) {
        val regParams = data.get("RegParameters")
        if (regParams != null) {
          val lambda = regParams.get("lambda").asInstanceOf[Double]
          if (lambda != 0) {
            return lambda
          }
        }
      }
    }
    return Functions.DEFAULT_LABMDA
  }


  // Data generation function for linear regression.
  // Function forked from spark-perf project.

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
