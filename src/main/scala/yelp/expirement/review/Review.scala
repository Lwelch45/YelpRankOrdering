package yelp.expirement.review

import java.nio.file.{Paths, Files}
import java.util.Random

import org.apache.spark.SparkContext
import org.apache.spark.mllib.clustering
import org.apache.spark.mllib.clustering.{LDA, KMeans, DistributedLDAModel, KMeansModel}
import org.apache.spark.mllib.regression.{LabeledPoint, RidgeRegressionWithSGD, RidgeRegressionModel, LinearRegressionModel}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SQLContext}
import topicmodeling.regulaizers.{SymmetricDirichletTopicRegularizer, SymmetricDirichletDocumentOverTopicDistributionRegularizer}
import topicmodeling.{RobustPLSA, TokenEnumerator}
import yelp.{Util}
import yelp.expirement.RankingExperiment
import org.apache.spark.mllib.linalg.{Vector, Vectors}

import scala.collection


/**
 * This class takes a business id and a trained
 * user-score model and trains a KMeans cluster model
 * on the reviews for the given business id.
 * Created by laurencewelch on 4/15/15.
 */
class Review(@transient protected val sc: SparkContext, @transient protected val sqlContext: SQLContext,
             protected val businessID: String, protected val K: Int)
  extends RankingExperiment{

  var usefulModel: RidgeRegressionModel = null
  var TFClusterModel: KMeansModel = null
  var LDAClusterModel: DistributedLDAModel = null


  var TFClusterError: Double = 0.0

  val businessReviews = Util.vectorize(sqlContext.sql(s"""
        SELECT r.text
        FROM review r
        WHERE r.business_id = '$businessID'
      """)).rdd.map(row => row(2).asInstanceOf[Vector] )

  var processedReviews: RDD[(Long, Vector)] = null

  override def listCentroids(): Array[Vector] = {
    Array.empty
  }

  override def perform(): Unit = {}

  def observe(): String = {
    val reviews = Util.vectorize(sqlContext.sql(s"""
        SELECT r.text
        FROM review r
        WHERE r.business_id = '$businessID'
      """)).rdd.map(row => (row(0).asInstanceOf[String], row(2).asInstanceOf[Vector])).cache()
    val usefulPredictions = usefulModel.predict(reviews.map(row => row._2))

    reviews.zip(usefulPredictions)
    reviews.zipWithIndex()

    reviews.zip(TFClusterModel.predict(reviews.map(row => row._2)))
    ""
  }

  override def train(): Unit = {
    trainTF()
    trainPLSA()
    trainUseful()

  }

  def trainUseful(): Unit = {

    var reviews = sqlContext.sql("""
        SELECT text, votes.useful as label
        FROM review
                                 """)
    reviews = Util.vectorize(reviews)
    val data = reviews.rdd.map(row =>LabeledPoint(row.getLong(1), row(3).asInstanceOf[Vector]))
    val (trainingData, testingData) = Util.splitData(data)

    usefulModel = RidgeRegressionWithSGD.train(trainingData, 12)

    val prediction = usefulModel.predict(testingData.map(_.features))
    val predictionAndLabel = prediction.zip(testingData.map(_.label))

    val loss = predictionAndLabel.map { case (p, l) =>
      val err = p - l
      err * err
    }.reduce(_ + _)
    val rmse = math.sqrt(loss / testingData.count())
    val mse = predictionAndLabel.map{case(v, p) => math.pow((v - p), 2)}.mean()
    println(s"Test RMSE = $rmse.")


    try{
      println("RidgeRegression MSE: " + mse)
      println("RidgeRegression RMSE: " + rmse)
    } catch {
      case e: Exception => println("exception caught: " + e)
    }
  }

  def trainTF(): Unit = {

    TFClusterModel = KMeans.train(businessReviews, K, 10)
    TFClusterError = TFClusterModel.computeCost(businessReviews)

    //TODO store vectorized reviews?
    //(model, reviews, reviewRDD, WSSSE)
  }

  def trainPLSA(): Unit = {
    val numberOfTopics = K
    val numberOfIterations = 10
    processedReviews = businessReviews.zipWithIndex().map((row) => (row._2, row._1))
    // Set LDA parameters
    val lda = new LDA().setK(numberOfTopics).setMaxIterations(numberOfIterations)
    LDAClusterModel = lda.run(processedReviews)
  }
}
