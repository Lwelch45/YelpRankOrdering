package yelp.expirement

import org.apache.spark.mllib.linalg.{Vector, Vectors}

/**
 * Created by laurencewelch on 4/13/15.
 */
abstract class ClusteringExperiment extends Expirement {
  def listCentroids(): Array[Vector]

}
