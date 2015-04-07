name := "YelpRankOrdering"

version := "1.0"

scalaVersion := "2.11.6"

libraryDependencies ++= Seq(
  "org.apache.spark" % "spark-core_2.10" % "1.3.0",
  "org.apache.spark" % "spark-sql_2.10" % "1.3.0",
  "org.apache.spark" % "spark-mllib_2.10" % "1.3.0"
)