import numpy as np
from pyspark import SparkContext


class KMeansPlusPlus:
    def __init__(self, k, max_iterations=100, tolerance=1e-5):
        """
        Initializes the KMeansPlusPlus class.

        Args:
            k: Number of clusters.
            max_iterations: Maximum number of iterations.
            tolerance: Convergence threshold for centroid shift.
        """
        self.k = k
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.centroids = None
        self.labels = None
        self.centroids_history = []
        self.labels_history = []

    def euclidean_distance(self, point1, point2):
        """Computes the Euclidean distance between two points."""
        return np.linalg.norm(np.array(point1) - np.array(point2))

    def initialize_centroids(self, data):
        """Initializes centroids using K-Means++ method."""
        centroids = [data[np.random.randint(len(data))]]  # First random centroid

        for _ in range(1, self.k):
            distances = np.array([min(self.euclidean_distance(x, c)**2 for c in centroids) for x in data])
            probabilities = distances / distances.sum()
            cumulative_probs = np.cumsum(probabilities)
            r = np.random.rand()
            next_centroid_index = np.searchsorted(cumulative_probs, r)
            centroids.append(data[next_centroid_index])

        return np.array(centroids)

    def fit(self, data):
        """
        Runs the K-Means++ algorithm on the provided data.

        Args:
            data: Dataset as a NumPy array.

        Returns:
            centroids: Final list of centroids.
            cluster_labels: List of cluster assignments for each point.
        """
        # Initialize SparkContext
        sc = SparkContext("local", "KMeans++")

        # Step 1: Initialize centroids
        self.centroids = self.initialize_centroids(data)

        for iteration in range(self.max_iterations):
            # Step 2: Assign each point to the nearest centroid
            rdd_data = sc.parallelize(data)
            cluster_assignments = rdd_data.map(
                lambda point: (np.argmin([self.euclidean_distance(point, c) for c in self.centroids]), point)
            )

            # Step 3: Calculate new centroids
            new_centroids = (
                cluster_assignments.groupByKey()
                .mapValues(lambda points: np.mean(list(points), axis=0))
                .collect()
            )

            # Sort by cluster label and get the new centroids
            new_centroids = np.array([centroid[1] for centroid in sorted(new_centroids, key=lambda x: x[0])])
            
            self.centroids_history.append(new_centroids.copy())
            self.labels_history.append(cluster_assignments.map(lambda x: x[0]).collect())

            # Step 4: Check for convergence
            centroid_shift = np.linalg.norm(new_centroids - self.centroids)
            print(f"Iteration {iteration + 1}: Total centroid shift = {centroid_shift:.4f}")
            if centroid_shift < self.tolerance:
                print(f"Convergence reached after {iteration + 1} iterations.")
                break

            self.centroids = new_centroids

        # Step 5: Final cluster assignment
        self.labels = rdd_data.map(
            lambda point: np.argmin([self.euclidean_distance(point, c) for c in self.centroids])
        ).collect()

        sc.stop()  # Stop SparkContext

        return self.centroids, self.labels, self.labels_history, self.centroids_history



