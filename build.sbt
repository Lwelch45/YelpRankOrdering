name := "YelpRankOrdering"

version := "1.0"

scalaVersion := "2.10.5"

libraryDependencies ++= Seq(
  "org.apache.spark" % "spark-core_2.10" % "1.3.0",
  "org.apache.spark" % "spark-sql_2.10" % "1.3.0",
  "org.apache.spark" % "spark-mllib_2.10" % "1.3.0",
  "org.scalanlp" %% "breeze" % "0.8.1",
  "org.scalanlp" %% "breeze-natives" % "0.8.1"
)

resolvers ++= Seq(
  "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"
)

resolvers +=
  "Sonatype OSS Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots"