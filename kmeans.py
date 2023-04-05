
import matplotlib.pyplot as plt
import random
import numpy as np
import math


# create data 

# randomly initalize centroids 
# calculate distance from each data point to centroid
# assign those points to that centroid 
# move the centroids to the mean of data points it was assigned to

data = np.random.normal(2, 1, size=(50, 2))
data = np.append(data, np.random.normal(10, 1, size=(50, 2)), axis=0)

class KMeans:

    def __init__(self, data, k=2):
        self.data = data
        self.k = k
        self.centroids = np.array([random.choice(data) for _ in range(self.k)])
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
        for epoch in range(epochs):
            print("Epoch: ")
                
            for point in data:
                nearest = self.nearest_centroid(point)

                if nearest == None:
                    continue

                self.groups[nearest] = np.append(self.groups[nearest], np.array([point]), axis=0)

            # calculate mean for groups 
            for centroid_index, points in self.groups.items():
                x, y = np.mean(points[:, 0]), np.mean(points[:, 1])
            

model = KMeans(data=data, k=2)

model.fit()


plt.scatter(data[:, 0], data[:, 1])
plt.show()
