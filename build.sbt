name := "distributed ML-benchmark"

version := "0.1"

scalaVersion := "2.10.4"

parallelExecution in Test := false

{
  val excludeHadoop = ExclusionRule(organization = "org.apache.hadoop")
  libraryDependencies ++= Seq(
    "org.scalatest" %% "scalatest" % "1.9.1" % "test",
    "org.apache.spark" % "spark-core_2.10" % "1.3.1" excludeAll(excludeHadoop),
    "org.apache.spark" % "spark-mllib_2.10" % "1.3.1" excludeAll(excludeHadoop),
    "org.apache.spark" % "spark-sql_2.10" % "1.3.1" excludeAll(excludeHadoop),
    "org.apache.commons" % "commons-compress" % "1.7",
    "commons-io" % "commons-io" % "2.4",
    "org.scalanlp" % "breeze_2.10" % "0.11.2",
    "com.github.fommil.netlib" % "all" % "1.1.2" pomOnly(),
    "com.github.scopt" %% "scopt" % "3.3.0"
  )
}

{
  val defaultHadoopVersion = "1.0.4"
  val hadoopVersion =
    scala.util.Properties.envOrElse("SPARK_HADOOP_VERSION", defaultHadoopVersion)
  libraryDependencies += "org.apache.hadoop" % "hadoop-client" % hadoopVersion
}

resolvers ++= Seq(
  "Local Maven Repository" at Path.userHome.asFile.toURI.toURL + ".m2/repository",
  "Typesafe" at "http://repo.typesafe.com/typesafe/releases",
  "Spray" at "http://repo.spray.cc"
)