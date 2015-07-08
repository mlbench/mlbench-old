package MLbenchmark
import org.apache.spark.{SparkContext, SparkConf}
import org.apache.spark.mllib.regression
import MLbenchmark.solvers._
import MLbenchmark.utils._

object driver {
	def main(args: Array[String]) {
		val sc = new SparkContext("local","MLbenchmark","/Users/mac/Desktop/summer_project/spark-1.4.0/bin/",List("target/scala-2.10/driver_2.10-1.0.jar"))

		val train_file = "data/small_train.dat" 
		val test_file = "data/small_test.dat"
		//read in
		val train_data = OptUtils.loadLIBSVMData(sc, train_file, 4, 9947).cache()
		val test_data = OptUtils.loadLIBSVMData(sc, test_file, 4, 9947).cache()
		//SGD.run_SGD(train_data, test_data)
		Lbfgs.run_LBFGS(train_data, test_data)
		sc.stop
	}
}