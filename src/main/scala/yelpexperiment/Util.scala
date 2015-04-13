package yelpexperiment

import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.rdd.RDD

/**
 * Created by laurencewelch on 4/8/15.
 */
object Util {

  def splitData(data: RDD[LabeledPoint]): (RDD[LabeledPoint], RDD[LabeledPoint]) ={
    var splits = data.randomSplit(Array(.7,.3))
    (splits(0), splits(1))
  }

}
