import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from knn import KNN

iris = datasets.load_iris()
x, y = iris.data, iris.target
xtr, xte, ytr, yte = train_test_split(x, y, test_size=0.2, random_state=1234)

plt.scatter(x[:, 2], x[:, 3])
plt.show()

uu = KNN(k=5)
uu.fit(xtr, ytr)
pred = uu.predict(xte)
print(pred)
acc = np.sum(pred == yte) / len(yte)
print(acc)
