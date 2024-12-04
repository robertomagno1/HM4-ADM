import numpy as np
from pyspark import SparkContext
import warnings


class KmeansSpark:
    def __init__(self, k, max_iters=100, tol=1e-6, app_name="KMeans Spark", track_history=False):
        """
        Initializes the KMeans class.

        Args:
            k: Number of clusters.
            max_iters: Maximum number of iterations.
            tol: Convergence tolerance.
            app_name: Spark application name.
            track_history: If True, track centroid and label history for visualization.
        """
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.app_name = app_name
        self.track_history = track_history
        self.centroids = None
        self.clustered_data = None
        self.centroids_history = []
        self.labels_history = []

    def fit(self, data):
        """
        Runs the K-Means algorithm on the provided data.

        Args:
            data: Input data as Pandas DataFrame or list of lists.

        Returns:
            centroids: Final centroids.
            clustered_data: Data grouped by clusters.
        """
        # Suppress specific warnings
        warnings.filterwarnings('ignore', category=RuntimeWarning)

        # Spark context configuration
        sc = SparkContext("local", self.app_name)

        try:
            # Convert data to Spark RDD
            if not isinstance(data, list):
                rdd_data = sc.parallelize(data.values.tolist())
            else:
                rdd_data = sc.parallelize(data)

            # Step 1: Random initialization of centroids
            centroids = rdd_data.takeSample(False, self.k)

            for iteration in range(self.max_iters):
                # Mapping: Assign each point to the nearest cluster
                clustered_points = (
                    rdd_data.map(
                        lambda point: (
                            np.argmin([np.linalg.norm(np.array(point) - np.array(centroid)) for centroid in centroids]),
                            point
                        )
                    ).groupByKey()
                    .mapValues(list)
                )

                # Reduction: Calculate new centroids
                new_centroids = (
                    clustered_points.mapValues(lambda points: np.mean(points, axis=0))
                                    .collectAsMap()
                )
                new_centroids = [new_centroids[i] for i in range(self.k)]

                # Track history if required
                if self.track_history:
                    # Collect cluster labels for each point
                    current_labels = rdd_data.map(
                        lambda point: np.argmin([np.linalg.norm(np.array(point) - np.array(centroid)) for centroid in centroids])
                    ).collect()

                    self.centroids_history.append(centroids)
                    self.labels_history.append(current_labels)

                # Calculate total centroid shift
                shift = sum(
                    np.linalg.norm(np.array(new_centroids[i]) - np.array(centroids[i]))
                    for i in range(self.k)
                )
                print(f"Iteration {iteration + 1}: total shift = {shift:.4f}")

                if shift < self.tol:
                    print(f"Convergence reached after {iteration + 1} iterations.")
                    break

                # Update centroids
                centroids = new_centroids

            # Final output: Reassign points to clusters
            clustered_data = rdd_data.map(
                lambda point: (
                    np.argmin([np.linalg.norm(np.array(point) - np.array(centroid)) for centroid in centroids]),
                    point
                )
            ).collect()

            self.centroids = centroids
            self.clustered_data = clustered_data

        finally:
            # Close Spark context
            sc.stop()

        return self.centroids, self.clustered_data

    def predict(self, data):
        """
        Assigns new data points to the nearest cluster.

        Args:
            data: New data points as a NumPy array or list.

        Returns:
            List of cluster assignments for each data point.
        """
        if self.centroids is None:
            raise ValueError("The model has not been trained yet. Call fit() first.")
        
        return [
            np.argmin([np.linalg.norm(np.array(point) - np.array(centroid)) for centroid in self.centroids])
            for point in data
        ]