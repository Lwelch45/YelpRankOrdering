package yelp

import org.apache.spark.SparkContext
import org.apache.spark.mllib.clustering.KMeansModel
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import scala.util.Sorting

object IDDistanceOrdering extends Ordering[((Vector,Long), Double)] {
  def compare(a:((Vector,Long), Double), b:((Vector,Long), Double)) = a._2 compare b._2
}
object DistanceOrdering extends Ordering[(Vector, Double)] {
  def compare(a:(Vector, Double), b:(Vector, Double)) = a._2 compare b._2
}

/**
 * Created by laurencewelch on 4/14/15.
 */
class KMeansClusterModel(@transient val sc: SparkContext,
                         @transient val model: KMeansModel) {

  val centers = sc.parallelize(model.clusterCenters).zipWithIndex()

  /**
   *
   * @param point the point to check
   * @param N N
   * @return the Nth furthest cluster
   */
  def furthestNthCluster(point: Vector, N: Int = 1): ((Vector,Long), Double) = {
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
  def closestNthPointToCentroid(points: RDD[Vector], clusterID: Long, N: Int = 0): (Vector, Double) = {
    val center = centers.filter((vec) => {
      vec._2 == clusterID
    }).first()._1

    points.map((vec) => {
      (vec, Vectors.sqdist(center, vec))
    }).top(N)(DistanceOrdering).apply(N-1)
  }


}
