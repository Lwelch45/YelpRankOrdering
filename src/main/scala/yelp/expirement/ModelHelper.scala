package yelp.expirement

import org.apache.spark.mllib.linalg.{Vectors, Vector}
import org.apache.spark.mllib.regression.LinearRegressionModel
import org.apache.spark.rdd.RDD

/**
 * Created by laurencewelch on 5/5/15.
 */

object ModelHelper {

  /**
   *
   * @param point the point to check
   * @param N N
   * @return the Nth furthest cluster
   */
  def furthestNthCluster(centers: RDD[(Vector, Long)], point: Vector, N: Int = 1): ((Vector,Long), Double) = {
    assert(N <= centers.count())

    centers.map((vec) => {
      (vec, Vectors.sqdist(vec._1, point))
    }).top(N)(IDDistanceOrdering).reverse.apply(N-1)
  }

  /**
   *
   * @param points the points to check
   * @param clusterID the id of the centroid to check against
   * @param N N
   * @return the Nth closest point to the specified centroid
   */
  def closestNthPointToCentroid(centers: RDD[(Vector, Long)], points: RDD[Vector], clusterID: Long, N: Int = 0): (Vector, Double) = {
    assert(N <= points.count())
    val center = centers.filter((vec) => {
      vec._2 == clusterID
    })

    assert(center.count() == 1, "The center was not found given the clusterID")

    points.map((vec) => {
      (vec, Vectors.sqdist(center.first()._1, vec))
    }).top(N)(DistanceOrdering).apply(N-1)
  }

  def getMostUseful(reviews: RDD[Vector], usefulModel: LinearRegressionModel): (Vector, Double) = {
    var max: Vector = null
    var mostUseful = Double.MinValue //usefulness can never be negative

    reviews.foreach(review => {
      val useful = usefulModel.predict(review)
      if (useful > mostUseful){
        mostUseful = useful
        max = review
      }
    })
    (max, mostUseful)
  }
}

object IDDistanceOrdering extends Ordering[((Vector,Long), Double)] {
  def compare(a:((Vector,Long), Double), b:((Vector,Long), Double)) = a._2 compare b._2
}
object DistanceOrdering extends Ordering[(Vector, Double)] {
  def compare(a:(Vector, Double), b:(Vector, Double)) = a._2 compare b._2
}
