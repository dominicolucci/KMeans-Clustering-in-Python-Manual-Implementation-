import numpy as np

class KMeans:
    def __init__(self, k=3, num_iter=1000, order=2):
        """
        Initializes the KMeans class.
        
        Args:
            k (int): Number of clusters.
            num_iter (int): Maximum number of iterations.
            order (int): Distance metric order (1=Manhattan, 2=Euclidean).
        """
        np.random.seed(42)
        self.k = k
        self.num_iter = num_iter
        self.centers = None

        if order in [1, 2]:
            self.order = order
        else:
            raise Exception("Unknown Order")     

    def fit(self, X):
        """
        Fits the KMeans model to the input data X.
        
        Args:
            X (ndarray): Input data of shape (m, n).
        """
        m, n = X.shape
        self.centers = np.zeros((self.k, n))
        self.cluster_idx = np.zeros(m)

        # Initialize cluster centers randomly between 10th and 90th percentiles
        for i in range(self.k):
            for j in range(n):
                low = np.percentile(X[:, j], 10)
                high = np.percentile(X[:, j], 90)
                self.centers[i, j] = np.random.uniform(low, high)

        for i in range(self.num_iter):
            new_centers = np.zeros((self.k, n))

            # Compute distances and assign each point to the nearest cluster
            distances = np.linalg.norm(X[:, np.newaxis] - self.centers, ord=self.order, axis=2)
            cluster_idx = np.argmin(distances, axis=1)

            # Update cluster centers
            for idx in range(self.k):
                cluster_points = X[cluster_idx == idx]
                if len(cluster_points) > 0:
                    if self.order == 2:
                        new_centers[idx] = np.mean(cluster_points, axis=0)
                    elif self.order == 1:
                        new_centers[idx] = np.median(cluster_points, axis=0)

            # Early stopping if clusters do not change
            if np.all(cluster_idx == self.cluster_idx):
                print(f"Early stopped at iteration {i}")
                break

            self.centers = new_centers
            self.cluster_idx = cluster_idx

        return self

    def predict(self, X):
        """
        Predicts the closest cluster for each sample in X.
        
        Args:
            X (ndarray): Input data of shape (m, n).
        
        Returns:
            ndarray: Cluster index for each input sample.
        """
        distances = np.linalg.norm(X[:, np.newaxis, :] - self.centers, axis=2, ord=self.order)
        return np.argmin(distances, axis=1)
