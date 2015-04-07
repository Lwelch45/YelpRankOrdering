

import org.apache.spark.mllib.classification.LogisticRegressionWithSGD
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.SQLContext
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.mllib.regression.LinearRegressionWithSGD
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.sql.functions._
import org.apache.spark.mllib.linalg.{Vector, Vectors}

/**
 * Created by laurencewelch on 4/5/15.
 */
object Main {
  def main(args: Array[String]) {
    val sparkConf = new SparkConf()
      .setAppName("YelpDataSetChallenge")
      .setMaster("mesos://10.132.51.174:5050/mesos")

    val sc = new SparkContext(sparkConf)
    val sqlC = new SQLContext(sc)

    val bf = sqlC.jsonFile("hdfs://10.132.51.175/yelp/yelp_academic_dataset_business.json")

    var rf = sqlC.jsonFile("hdfs://10.132.51.175/yelp/yelp_academic_dataset_review.json")


    //collect statistics on review counts
    val bfRows = bf.rdd
    val rows = bfRows.map(row => {
      (row.getLong(11))
    })
    val count = rows.count()
    val mean = rows.sum / count
    val devs = rows.map(score => (score - mean) * (score - mean))
    val stddev = Math.sqrt(devs.sum / count)

    bf.registerTempTable("business")
    rf.registerTempTable("review")

    var reviews = sqlC.sql("""
        SELECT text, votes.useful as label
        FROM review
                           """)

    val tokenizer = new Tokenizer()
      .setInputCol("text")
      .setOutputCol("words")

    val hashingTF = new HashingTF()
      .setNumFeatures(1000)
      .setInputCol(tokenizer.getOutputCol)
      .setOutputCol("features")
    reviews = tokenizer.transform(reviews);
    reviews = hashingTF.transform(reviews);

    var data = reviews.rdd.map(row =>{
      LabeledPoint(row.getLong(1), row(3).asInstanceOf[Vector])
    })
    var splits = data.randomSplit(Array(.7,.3))
    var (trainingData, testingData) = (splits(0), splits(1))
    var model = LogisticRegressionWithSGD.train(trainingData, 12)

    val prediction = model.predict(testingData.map(_.features))
    val predictionAndLabel = prediction.zip(testingData.map(_.label))

    val loss = predictionAndLabel.map { case (p, l) =>
      val err = p - l
      err * err
    }.reduce(_ + _)
    val rmse = math.sqrt(loss / testingData.count())

    println(s"Test RMSE = $rmse.")

    sc.stop()

    //var splits = reviews.rdd.randomSplit(Array(0.7, 0.3))
    //var (preTrainingData, preTestData) = (splits(0), splits(1))
    //var trainingData = preTrainingData.map(row => {
    //  (row.getString(0), (row.getInt(1) /100).toDouble)
    //})
    //var testingData = preTestData.map(row => {
    //  (row.getString(0), (row.getInt(1) /100).toDouble)
    //})
  }
}
