package MLbenchmark.solvers

import java.io.{FileWriter, BufferedWriter, PrintWriter}

import org.apache.spark.SparkContext
import org.apache.spark.mllib.evaluation.BinaryClassificationMetrics
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.util.{LinearDataGenerator, MLUtils}
import org.apache.spark.mllib.regression.{RidgeRegressionWithSGD, RidgeRegressionModel}
import org.apache.spark.mllib.optimization._
import org.apache.spark.rdd.RDD
import MLbenchmark.utils._
import breeze.linalg.DenseVector
import org.apache.spark.sql.DataFrame
import scala.io.Source
import scala.collection.mutable.ArrayBuffer
import breeze.optimize.{CachedDiffFunction, DiffFunction, OWLQN}
import org.apache.spark.ml.regression._
import breeze.linalg.{DenseVector => BDV}
import org.apache.spark.ml.regression.LinearRegression

object Lbfgs{
  //lossType = 0: SVM, losstype=1: ridge regression, losstype=2 logistic regression with L2-regularzation
    def run_LBFGS(trainData: RDD[LabeledPoint], testData: RDD[LabeledPoint],dataset: DataFrame, maxNumIterations:Int, chkptIter:Int, optimalVal:Double, lossType:Int) {

      val numFeatures = trainData.take(1)(0).features.size

      // Run training algorithm to build the model
      val numCorrections = 20
      val convergenceTol = 0
      val regParam = 0.001
      var initialWeights = Vectors.dense(new Array[Double](numFeatures))

      val training = trainData.map(point => (point.label, Vectors.sparse(numFeatures, point.features.index, point.features.data)))

      val chkptNum = math.floor(maxNumIterations / chkptIter).toInt
      var totalTime: Long = 0

      if(lossType!=2) {
        var gradient: Gradient =
          lossType match {
            case 0 => new HingeGradient()
            case 1 => new LeastSquaresGradient()
            case 3 => new LogisticGradient()
            case 4 => new LogisticGradient()
          }
        var updater =
          lossType match {
            case 0 => new SquaredL2Updater()
            case 1 => new SquaredL2Updater()
            case 3 => new SquaredL2Updater()
            case 4 => new L1Updater()
          }
        var startTime = System.nanoTime()
        val (weights, objectiveValue) = LoggingLbfgs.runLBFGS(
          training,
          gradient,
          updater,
          numCorrections,
          0,
          maxNumIterations,
          0.001,
          initialWeights)
        objectiveValue.foreach(println)
        recordLBFGS(optimalVal, maxNumIterations, chkptIter, objectiveValue, startTime)
      }
      else {
        val trainer = (new LoggingLinearRegression).setElasticNetParam(1.0).setRegParam(0.001).setMaxIter(400)
        val model = trainer.fit(dataset)

        //lbfgs for lasso regression(binary)
//        val lbfgs = new OWLQN[Double,DenseVector[Double]](maxIter =100,m = 1,l1reg = 1.0,tolerance=1.0E-8)

//        def objectValue(init:DenseVector[Double]) = {
//          val f = new DiffFunction[DenseVector[Double]]{
//            def calculate(x: DenseVector) = {
//              (Double,)
//            }
//          }
//        }
//        lbfgs.iterations(new CachedDiffFunction(costFun), initialWeights.toBreeze.toDenseVector)
      }
    }


    private def recordLBFGS(optimalVal:Double,maxIter:Int, chkptIter:Int,loss:Array[Double],startTime: Long) = {

      var iter = 0
      var chkptNum = maxIter/chkptIter
      var timeHistory = new ArrayBuffer[Double](maxIter)
      for(line <- Source.fromFile("output/time_Lbfgs.txt").getLines()){
        timeHistory.append(line.toDouble)
      }
      val name = "Lbfgs"
      val totalIter = loss.length
      for( iter <- 1 to chkptNum) {
        // find updates to alpha, w
        val iterNum = iter * chkptIter
        var primalObjective: Double = 0.0
        var time:Long = 1
        if(totalIter <= iterNum) {
          primalObjective= loss(totalIter - 1)
          time = (timeHistory(totalIter - 1) - startTime).toLong
        }
        else {
          primalObjective= loss(iterNum - 1)
          time = (timeHistory(iterNum - 1) - startTime).toLong
        }
        val subObjective:Double = primalObjective - optimalVal
        var pw = new PrintWriter(new BufferedWriter(new FileWriter("output/Iter_" +name+".txt", true)))

        pw.println(iterNum)
        pw.flush()
        pw.close

        pw = new PrintWriter(new BufferedWriter(new FileWriter("output/sub_" +name+".txt", true)))
        pw.println(subObjective)
        pw.flush()
        pw.close

        pw = new PrintWriter(new BufferedWriter(new FileWriter("output/time_" +name+".txt", true)))
        pw.println(time/1e6)
        pw.flush()
        pw.close
        println("PrimalObjective: "  + primalObjective)
        println("subObjective" + subObjective)
        println("iterations: " + iterNum)
        //println("time: " + time + "ms")
      }
    }

}


