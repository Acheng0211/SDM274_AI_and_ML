import numpy as np
from scipy.spatial import KDTree
from collections import Counter

class kNN:
    def __init__(self, k = 3):
        self.k = k
        self.kdtree = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        self.kdtree = KDTree(X)

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        # calculate the distance between x and all samples in the training set
        distances = [np.linalg.norm(x - x_train) for x_train in self.X_train]
        # get k nearest samples, labels
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # majority vote, most common class label
        most_common = Counter(map(tuple, k_nearest_labels)).most_common(1)
        return most_common[0][0]