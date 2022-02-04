import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


iris = load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['label'] = iris.target
data = np.array(data.iloc[:100, :])
X, y = data[:, :-1], data[:, -1]
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25)

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(train_X, train_y)
print(clf.score(test_X, test_y))