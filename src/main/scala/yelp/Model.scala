package yelp

import java.nio.file.{Files, Paths}

import org.apache.spark.SparkContext
import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.mllib.linalg.Vector
import org.apache.spark.mllib.regression._
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.{DataFrame, SQLContext}

/**
 * Created by laurencewelch on 4/8/15.
 */
class Model(@transient protected val sc: SparkContext,
            @transient protected val sqlContext: SQLContext,
            tokenizer: Tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words"),
            hashingTF: HashingTF = new HashingTF().setNumFeatures(1000).setInputCol("words").setOutputCol("features")) {
  val ridgeModelFileName = "ridgeModel.save"
  val linearModelFileName = "linearModel.save"

  def testLinearRegressionSGD(trainingData: RDD[LabeledPoint] ,testingData: RDD[LabeledPoint]): (LinearRegressionModel, (Double,Double) ) ={
    var model = LinearRegressionWithSGD.train(trainingData, 12)

    val prediction = model.predict(testingData.map(_.features))
    val predictionAndLabel = prediction.zip(testingData.map(_.label))

    val loss = predictionAndLabel.map { case (p, l) =>
      val err = p - l
      err * err
    }.reduce(_ + _)
    val rmse = math.sqrt(loss / testingData.count())
    val MSE = predictionAndLabel.map{case(v, p) => math.pow((v - p), 2)}.mean()
    println(s"Test RMSE = $rmse.")
    (model,(rmse,MSE))
  }

  def testRidgeRegressionSGD(trainingData: RDD[LabeledPoint] ,testingData: RDD[LabeledPoint]): (RidgeRegressionModel, (Double,Double) )={
    var model = RidgeRegressionWithSGD.train(trainingData, 12)

    val prediction = model.predict(testingData.map(_.features))
    val predictionAndLabel = prediction.zip(testingData.map(_.label))

    val loss = predictionAndLabel.map { case (p, l) =>
      val err = p - l
      err * err
    }.reduce(_ + _)
    val rmse = math.sqrt(loss / testingData.count())
    val MSE = predictionAndLabel.map{case(v, p) => math.pow((v - p), 2)}.mean()
    println(s"Test RMSE = $rmse.")
    (model,(rmse,MSE))
  }

  def generateUsefulModels(sc: SparkContext, reviews: DataFrame): (LinearRegressionModel, RidgeRegressionModel) ={
    var data = reviews.rdd.map(row =>{
      LabeledPoint(row.getLong(1), row(3).asInstanceOf[Vector])
    })
    val (trainingData, testingData) = Util.splitData(data)

    val linearModel = testLinearRegressionSGD(trainingData,testingData)
    val ridgeModel = testRidgeRegressionSGD(trainingData, testingData)

    //try to save models
    try{
      linearModel._1.save(sc, linearModelFileName)
      println("LinearRegression MSE: " + linearModel._2._2)
      println("LinearRegression RMSE: " + linearModel._2._1)
    } catch {
      case e: Exception => println("exception caught: " + e)
    }

    try{
      ridgeModel._1.save(sc, ridgeModelFileName)
      println("RidgeRegression MSE: " + ridgeModel._2._2)
      println("RidgeRegression RMSE: " + ridgeModel._2._1)
      (linearModel._1, ridgeModel._1)
    } catch {
      case e: Exception => println("exception caught: " + e)
        (linearModel._1, null)
    }
  }

  def computeBusinessStats(businessFrame: DataFrame): Unit ={
    //collect statistics on review counts
    val bfRows = businessFrame.rdd
    val rows = bfRows.map(row => {
      (row.getLong(11))
    })
    val count = rows.count()
    val mean = rows.sum / count
    val devs = rows.map(score => (score - mean) * (score - mean))
    val stddev = Math.sqrt(devs.sum / count)
  }

  def transformReviewDF(reviews: DataFrame): DataFrame ={
    hashingTF.transform(tokenizer.transform(reviews))
  }

  def trainUseful(sc: SparkContext, sqlContext: SQLContext): (LinearRegressionModel, RidgeRegressionModel) ={
    var reviews = sqlContext.sql("""
        SELECT text, votes.useful as label
        FROM review
                                 """)
    reviews = transformReviewDF(reviews)
    generateUsefulModels(sc, reviews)
  }

  def loadOrTrainModel(): (LinearRegressionModel,RidgeRegressionModel) ={
    var (linear,ridge): (LinearRegressionModel,RidgeRegressionModel) = null
    if(!Files.exists(Paths.get(linearModelFileName))){
      var (linearM, ridgeM) = trainUseful(sc, sqlContext)
      linear = linearM
      ridge = ridgeM
    }else{
      linear = LinearRegressionModel.load(sc, linearModelFileName)
      ridge = RidgeRegressionModel.load(sc, ridgeModelFileName)
    }
    (linear,ridge)
  }
}
