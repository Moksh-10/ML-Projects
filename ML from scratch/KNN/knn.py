import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k: int = 3):
        self.k = k

    def fit(self, x, y):
        self.xtr = x
        self.ytr = y

    def predict(self, x):
        return [self._predict(i) for i in x]

    def euclidean_dist(self, a, b):
        return np.sqrt(np.sum((a-b)**2))

    def _predict(self, x):
        dist = [self.euclidean_dist(x, xt) for xt in self.xtr]
        top_k = np.argsort(dist)[:self.k]
        labels = [self.ytr[i] for i in top_k]
        cc = Counter(labels).most_common()
        return cc[0][0]



aa = KNN
b = np.array([1, 2])
c = np.array([1, 6])
a = aa()
d = a.euclidean_dist(b, c)
print(d)
