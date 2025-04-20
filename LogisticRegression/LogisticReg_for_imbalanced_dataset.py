# Generate and plot a synthetic imbalanced classification dataset
from collections import Counter
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

# IImbalanced dataset
X, y = make_classification(n_samples=10000, n_classes=2, n_features=2, n_clusters_per_class=1,
                           n_redundant=0, weights=[0.99], random_state=10)
# print(X)
# print(Counter(y))
import seaborn as sns
import pandas as pd
sns.scatterplot(x=pd.DataFrame(X)[0], y=pd.DataFrame(X)[1], hue=y)
# plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

# Hyperparameter tuning
from sklearn.linear_model import LogisticRegression
logistic = LogisticRegression()
penalty = ['l1', 'l2', 'elasticnet']
c_values = [100, 10, 1, 0.1, 0.01]
solver = ['newton-cg', 'sag', 'saga', 'lbfgs', 'liblinear']
class_weight = [{0:w, 1:y} for w in [1,10,50,100] for y in [1,10,50,100]]

params = dict(penalty = penalty, C=c_values, solver=solver, class_weight=class_weight)
# print(params)

from sklearn.model_selection import GridSearchCV, StratifiedKFold
cv = StratifiedKFold()
grid = GridSearchCV(estimator=logistic, param_grid=params, cv=cv, scoring='accuracy')

grid.fit(X_train, y_train)
# print(grid.best_params_)
# print(grid.best_score_)

y_pred = grid.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
score = accuracy_score(y_test, y_pred)
print(score)

confu_matrix = confusion_matrix(y_test, y_pred)
print(confu_matrix)

class_report = classification_report(y_test, y_pred)
print(class_report)