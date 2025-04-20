import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#create synthetic dataset
from sklearn.datasets import make_classification
# X, y = make_classification(n_samples=1000, n_features=2, n_classes=2,
#                            n_clusters_per_class=1, n_redundant=0, random_state=15)
# print(X)
# print(y)

# sns.scatterplot(x= pd.DataFrame(X)[0], y= pd.DataFrame(X)[1], hue=y)
# plt.show()

# importing support vector classification
from sklearn.svm import SVC
# svc = SVC(kernel= 'linear')

from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)

# svc.fit(X_train, y_train) # training
# y_pred = svc.predict(X_test) # prediction

from sklearn.metrics import classification_report,confusion_matrix, accuracy_score
# print(classification_report(y_test, y_pred))
# print(confusion_matrix(y_test,y_pred))
# print(accuracy_score(y_test, y_pred))



X_rbf, y_rbf = make_classification(n_samples=1000, n_classes=2, n_features=2, n_redundant=0,
                           n_clusters_per_class=2)

# sns.scatterplot(x= pd.DataFrame(X_rbf)[0], y= pd.DataFrame(X_rbf)[1], hue=y)
# plt.show()
X_train, X_test, y_train, y_test = train_test_split(X_rbf, y_rbf, test_size=0.25, random_state=10)

rbf = SVC(kernel='rbf')
rbf.fit(X_train, y_train)
y_pred = rbf.predict(X_test)
print('*********************************   RBF    ***************************************')
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


poly = SVC(kernel='poly')
poly.fit(X_train, y_train)
y_pred = poly.predict(X_test)
print('*********************************   POLY    ***************************************')
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


sigmoid = SVC(kernel='sigmoid')
sigmoid.fit(X_train, y_train)
y_pred = sigmoid.predict(X_test)
print('***************************************   sigmoid   *********************************')
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


linear = SVC(kernel='linear')
linear.fit(X_train, y_train)
y_pred = linear.predict(X_test)
print('***************************************   Linear   *********************************')
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV
params = {"C": [0.1, 1, 10, 100, 1000],
          'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
          'kernel': ['rbf', 'linear', 'poly', 'sigmoid']
          }
grid = GridSearchCV(estimator=SVC(), param_grid=params, cv=5, refit=True, verbose=3)
grid.fit(X_train, y_train)
y_pred = grid.predict(X_test)
print('***************************************   HyperparameterTuning   *********************************')
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(grid.best_params_)
print(grid.best_score_)