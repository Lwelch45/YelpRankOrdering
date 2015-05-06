package yelp

import org.apache.spark.ml.feature.{HashingTF, Tokenizer}
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.DataFrame

/**
 * Created by laurencewelch on 4/8/15.
 */
object Util {

  def splitData(data: RDD[LabeledPoint]): (RDD[LabeledPoint], RDD[LabeledPoint]) ={
    var splits = data.randomSplit(Array(.7,.3))
    (splits(0), splits(1))
  }

  /**
   *
   * @param frame DataFrame containing column with text values
   * @return DataFrame containing vectorized features
   */
  def vectorize(frame: DataFrame): DataFrame = {
    val tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words")
    val hashingTF  = new HashingTF().setNumFeatures(1000).setInputCol("words").setOutputCol("features")
    hashingTF.transform(tokenizer.transform(frame))
  }

}
