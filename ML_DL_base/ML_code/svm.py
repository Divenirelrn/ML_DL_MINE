import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
from matplotlib import pyplot as plt

def create_data():
    iris_data = load_iris()
    df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
    df['label'] = iris_data.target
    data = np.array(df.iloc[:100, [0,1,-1]])
    for i in range(len(data)):
        if data[i][-1] == 0:
            data[i][-1] = -1
    return data[:, :2], data[:,-1]


X, y = create_data()
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.25)
plt.scatter(X[:50, 0], X[:50, 1], label=0)
plt.scatter(X[50:, 0], X[50:, 1], label=1)
plt.legend()
plt.show()

from sklearn.svm import SVC
model = SVC()
model.fit(train_X, train_y)
print(model.score(test_X, test_y))
