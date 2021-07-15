from sklearn.model_selection import train_test_split
from sklearn import datasets, svm
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

#使用sklearn库中自带的iris数据集作为示例
iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=0) #分割数据集

param_grid = [
  {'C': [1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
]

svm_model = svm.SVC()

clf = GridSearchCV(svm_model, param_grid, cv=5)
clf.fit(X_train, y_train)

svm_model = svm.SVC()

clf = GridSearchCV(svm_model, param_grid, cv=5)
clf.fit(X_train, y_train)

best_model = clf.best_estimator_
print(clf.best_params_)

y_pred = best_model.predict(X_test)
print('accuracy', accuracy_score(y_test, y_pred))

