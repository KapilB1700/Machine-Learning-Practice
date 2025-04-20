# ************ DT Postpruning **********
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

iris = load_iris()
# print(iris)

# Independent and dependent features
X = pd.DataFrame(iris['data'], columns=['sepal length', 'sepal width', 'petal length', 'petal width'])
y = iris['target']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)

# Apply decision tree classifier
from sklearn.tree import DecisionTreeClassifier
treeclassifier = DecisionTreeClassifier(max_depth=2)

treeclassifier.fit(X_train, y_train)
# Visualize decision tree
from sklearn import tree
tree.plot_tree(treeclassifier, filled=True)
# plt.show()

y_pred = treeclassifier.predict((X_test))

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# print(accuracy_score(y_test,y_pred))
# print(confusion_matrix(y_test,y_pred))
# print(classification_report(y_test,y_pred))


# ************** DT Pre pruning and Hyperparameter Tuning*****************
params = {
    'criterion': ['gini','entropy','logloss'],
    'splitter': ['random', 'best'],
    'max_depth': [1,2,3,4,5],
    'max_features': ['auto', 'sqrt', 'log2']
}

from sklearn.model_selection import GridSearchCV
treemodel = DecisionTreeClassifier()
grid = GridSearchCV(treemodel, param_grid= params, cv=5, scoring='accuracy')
import warnings
warnings.filterwarnings('ignore')
grid.fit(X_train, y_train)
# print(grid.best_params_)
# print(grid.best_score_)
y_pred = grid.predict(X_test)

print(accuracy_score(y_test,y_pred))
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test, y_pred))