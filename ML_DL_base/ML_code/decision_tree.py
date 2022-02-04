import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


def create_data():
    iris = load_iris()
    data = pd.DataFrame(iris.data, columns=iris.feature_names)
    data['label'] = iris.target
    data = np.array(data.iloc[:100, [0,1,-1]])

    return data[:,:2], data[:, -1]

X, y = create_data()
train_X, test_x, train_y, test_y = train_test_split(X, y, test_size=0.3)

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
import graphviz

clf = DecisionTreeClassifier()
clf.fit(train_X, train_y)
print(clf.score(test_x, test_y))

tree_pic = export_graphviz(clf, out_file="decisionMakeTree.pdf")
with open("decisionMakeTree.pdf") as f:
    graph = f.read()
graph = graphviz.Source(graph)
print(graph)