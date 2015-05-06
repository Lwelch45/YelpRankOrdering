package yelp

import org.apache.spark.sql.SQLContext
import org.apache.spark.{SparkConf, SparkContext}
import yelp.expirement.review.Review

/**
 * Created by laurencewelch on 4/5/15.
 */
object Main {

  def main(args: Array[String]) {
    val sparkConf = new SparkConf().setAppName("YelpDataSetChallenge")
      .set("fs.s3n.awsAccessKeyId","AKIAIPEMZJ3VZMJ3MRGA")
      .set("fs.s3n.awsSecretAccessKey","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")
      .set("spark.executor.uri", "https://s3.amazonaws.com/jimi-yelp/spark-1.3.0-bin-hadoop2.4.tgz")

    val sc = new SparkContext(sparkConf)
    val sqlContext = new SQLContext(sc)

    val bf = sqlContext.jsonFile("s3n://AKIAIPEMZJ3VZMJ3MRGA:Y/oLyxBKk/e9/es+7B+rQexqEZNrciVFscnKDCSX@jimi-yelp/yelp_academic_dataset_business.json")
    bf.registerTempTable("business")

    val rf = sqlContext.jsonFile("https://s3.amazonaws.com/jimi-yelp/yelp_academic_dataset_review.json")
    rf.registerTempTable("review")

    val reviewExperiment = new Review(sc,sqlContext,"", 5)
    reviewExperiment.train()

    sc.stop()
  }

}
