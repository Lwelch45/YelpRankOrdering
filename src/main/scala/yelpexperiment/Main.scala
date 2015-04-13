package yelpexperiment

import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}

/**
 * Created by laurencewelch on 4/5/15.
 */
object Main {

  def main(args: Array[String]) {
    val sparkConf = new SparkConf().setAppName("YelpDataSetChallenge")
      .setMaster("mesos://10.132.51.175:5050/mesos")
      .set("spark.executor.uri", "https://s3.amazonaws.com/jimi-yelp/spark-1.3.0-bin-hadoop2.4.tgz")

    val sc = new SparkContext(sparkConf)
    val sqlContext = new SQLContext(sc)

    val bf = sqlContext.jsonFile("hdfs://10.132.51.175/yelp/yelp_academic_dataset_business.json")
    bf.registerTempTable("business")

    val rf = sqlContext.jsonFile("hdfs://10.132.51.175/yelp/yelp_academic_dataset_review.json")
    rf.registerTempTable("review")

    val model = new Model(sc, sqlContext)
    val models =  model.loadOrTrainModel()

    sc.stop()
  }

}
