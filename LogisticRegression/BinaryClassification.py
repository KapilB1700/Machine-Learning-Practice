import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification

# Creating the dataset
X,y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=15)
X=pd.DataFrame(X)
# print(X,y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train,y_test = train_test_split(X,y, test_size=0.30, random_state=42)

# Model training
from sklearn.linear_model import LogisticRegression
logistic = LogisticRegression()

logistic.fit(X_train, y_train)
y_pred = logistic.predict(X_test)
# print(y_pred)

# y_pred_proba = logistic.predict_proba(X_test) # returns probability of points
# print(y_pred_proba)

from sklearn.metrics import accuracy_score, confusion_matrix,classification_report
score = accuracy_score(y_test, y_pred)
# print(score)

confu_matrix = confusion_matrix(y_test, y_pred)
# print(confu_matrix)

class_report = classification_report(y_test, y_pred)
# print(class_report)

# Hyperparameter tuning and cross validation
model = LogisticRegression()
penalty = ['l1','l2','elasticnet']
c_values = [100,10,1.0,0.1,0.01]
solver = ['lbfgs', 'liblinear', 'newton-cg', 'sag', 'saga']

params = dict(penalty=penalty, C= c_values, solver= solver)

# using GridsearchCV for hyperparameter tuning

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
cv = StratifiedKFold()
grid = GridSearchCV(estimator=model, param_grid=params, scoring='accuracy', cv= cv, n_jobs=-1)

# print(grid)
grid.fit(X_train, y_train)
# print(grid.get_params)
# print(grid.best_params_)
# print(grid.best_score_)

y_pred_grid = grid.predict(X_test)
score = accuracy_score(y_test, y_pred_grid)
# print(score)

confu_matrix = confusion_matrix(y_test, y_pred_grid)
# print(confu_matrix)

class_report = classification_report(y_test, y_pred_grid)
# print(class_report)

# Randomized search CV
from sklearn.model_selection import RandomizedSearchCV

randomcv = RandomizedSearchCV(estimator=model, param_distributions=params, cv=5, scoring='accuracy')
randomcv.fit(X_train, y_train)
# print(randomcv)
# print(randomcv.best_score_)
# print(randomcv.best_params_)

y_pred_randomcv = randomcv.predict(X_test)
score = accuracy_score(y_test, y_pred_randomcv)
print(score)

confu_matrix = confusion_matrix(y_test, y_pred_randomcv)
print(confu_matrix)

class_report = classification_report(y_test, y_pred_randomcv)
print(class_report)
