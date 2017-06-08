name := "bench-spark"

version := "1.0"

scalaVersion := "2.11.7"
//scalaVersion := "2.10.5"

libraryDependencies ++= Seq(
  "org.apache.spark"              %% "spark-core"              % "2.1.0" % "provided",
  "org.apache.spark"              %% "spark-mllib"             % "2.1.0" % "provided",
  "org.apache.spark"              %% "spark-sql"               % "2.1.0" % "provided",
  "com.github.scala-incubator.io" %% "scala-io-file"           % "0.4.3",
  "org.rogach"                    %% "scallop"                 % "2.1.1",
  "org.scala-lang.modules" % "scala-parser-combinators_2.11" % "1.0.5"
)
unmanagedBase := baseDirectory.value / "libs"

// META-INF discarding
assemblyMergeStrategy in assembly := {
  case PathList("META-INF", xs @ _*) => MergeStrategy.discard
  case x => MergeStrategy.first
}

crossScalaVersions  := Seq("2.11.7", "2.10.5")

libraryDependencies := {
  CrossVersion.partialVersion(scalaVersion.value) match {
    case Some((2, scalaMajor)) if scalaMajor >= 11 =>
      libraryDependencies.value :+ "org.scala-lang.modules" %% "scala-xml" % "1.0.1"
    case _ =>
      libraryDependencies.value
  }
}
