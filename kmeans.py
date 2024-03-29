
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs  
import random
import numpy as np
import math
import copy


# randomly initalize centroids 
# calculate distance from each data point to centroid
# assign those points to that centroid 
# move the centroids to the mean of data points it was assigned to

X, y = make_blobs(n_samples=50, centers=3, n_features=2)

class KMeans:

    def __init__(self, data, k=2):
        self.data = data
        self.k = k
        self.centroids = np.array([random.choice(data) for _ in range(k)])
        self.groups = {x: np.empty((0, 2)) for x in range(k)}

    @staticmethod
    def dist(a, b):
        return math.sqrt((a[0] - b[0]) ** 2 - (a[1] - a[1]) ** 2)

    def nearest_centroid(self, x):
        d = 1000
        nearest = None

        if x in self.centroids:
            return None 

        for i, centroid in enumerate(self.centroids):
            if d > self.dist(centroid, x):
                d = min(d, self.dist(centroid, x))
                nearest = i

        return nearest

    @staticmethod
    def nearest_point(x):
        d = 1000
        nearest = None

        for i, point in enumerate(self.data):
            if d > self.dist(centroid, x):
                d = min(d, self.dist(centroid, x))
                nearest = point
        return nearest

    def fit(self, epochs=1):

        data = copy.copy(self.data)

        for epoch in range(epochs):
            print("Epoch: ", epoch+1)
                
            for i, point in enumerate(data):

                nearest = self.nearest_centroid(point)

                if nearest == None:
                    continue

                self.groups[nearest] = np.append(self.groups[nearest], np.array([point]), axis=0)

            # calculate mean for groups 
            for centroid_index, points in self.groups.items():
                mean_centroid = np.array([np.mean(points[:, 0]), np.mean(points[:, 1])])

                # if (mean_centroid == self.centroids[centroid_index]).all():
                    # print("Found centroids")

                # replace the centroid with new centroid 
                self.centroids[centroid_index] = mean_centroid

            print("Centroids: ", self.centroids)

            self.groups_copy = copy.copy(self.groups)
            self.groups = {x: np.empty((0, 2)) for x in range(self.k)}

    def plot(self):
        for group in self.groups_copy.values():
            plt.scatter(group[:, 0], group[:, 1])

        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], marker='x', linewidths=2)
        plt.show()

model = KMeans(data=X, k=3)

model.fit(epochs=10)

model.plot()
