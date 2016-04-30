name := "bench-spark"

version := "1.0"

// scalaVersion := "2.11.7"
scalaVersion := "2.10.5"

libraryDependencies ++= Seq(
  "org.apache.spark"  %% "spark-core"              % "1.6.0" % "provided",
  "org.apache.spark"  %% "spark-mllib"             % "1.6.0" % "provided",
  "org.apache.spark"  %% "spark-sql"               % "1.6.0" % "provided",
  "org.rogach"        %% "scallop"                 % "1.0.0"
)

// META-INF discarding
assemblyMergeStrategy in assembly := {
  case PathList("META-INF", xs @ _*) => MergeStrategy.discard
  case x => MergeStrategy.first
}