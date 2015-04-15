package yelp

import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.mllib.clustering.{KMeans, KMeansModel}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.mllib.regression.LinearRegressionModel
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SQLContext}
/**
 * Created by laurencewelch on 4/8/15.
 */
class RankingExperiment(@transient protected val sc: SparkContext,
                        @transient protected val sqlContext: SQLContext, usefulModel: LinearRegressionModel,
                        minReviewCount: Int = 15,
                        tokenizer: Tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words"),
                        hashingTF: HashingTF = new HashingTF().setNumFeatures(1000).setInputCol("words").setOutputCol("features"),
                        k: Int = 5) {
  var businessSample : DataFrame = null;

  def trainKMean(sqlContext: SQLContext, businessID: String): (KMeansModel, DataFrame, RDD[Vector] , Double) ={
    var reviews = sqlContext.sql(s"""
        SELECT r.text
        FROM review r
        WHERE r.business_id = '$businessID'
      """)
    reviews = hashingTF.transform(tokenizer.transform(reviews))
    var reviewRDD = reviews.rdd.map(row => {
      row(2).asInstanceOf[Vector]
    })
    val model = KMeans.train(reviewRDD, 5, 10)
    val WSSSE = model.computeCost(reviewRDD)

    (model, reviews, reviewRDD,WSSSE)
  }

  def furthestDistance(centers: Array[Vector], point: Vector): (Vector, Double) ={
    var max: Vector = null
    var maxDist = 0.0
    centers.foreach(center =>{
      val dist = Vectors.sqdist(center, point)
      if (dist > maxDist){
        maxDist = dist
        max = center
      }
    })
    (max, maxDist)
  }

  def getMostUseful(reviews: RDD[Vector]): (Vector, Double)={
    var max: Vector = null
    var mostUseful = 0.0
    reviews.foreach(review => {
      val useful = usefulModel.predict(review)
      if (useful > mostUseful){
        mostUseful = useful
        max = review
      }
    })
    (max, mostUseful)
  }

  def closestPointInCentroid(center: Vector, points: Array[Vector]): (Vector, Double) ={
    var closest: Vector = null
    var minDist = Double.MaxValue
    points.foreach(point =>{
      val dist = Vectors.sqdist(point, center)
      if (dist < minDist){
        minDist = dist
        closest = point
      }
    })
    (closest, minDist)
  }

  def filterBusinesses(): Unit ={
    businessSample = sqlContext.sql(s"""
        SELECT * business WHERE review_count > $minReviewCount
      """)
  }

  def getInstance(): Unit ={
    val business = businessSample.rdd.randomSplit(Array(.2,.8))(0).first()
    var modelData = trainKMean(sqlContext, business.getString(1))
    val (mostUseful, score) = getMostUseful(modelData._3)


  }

}
