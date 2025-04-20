# ************************* KNN Classification ****************************

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
'''
X, y = make_classification(n_samples=1000, n_features=3, n_classes=2, n_redundant=1, random_state=999)
# sns.scatterplot(x= pd.DataFrame(X)[0], y= pd.DataFrame(X)[1], hue=y)
# plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5, algorithm='auto')

classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test,y_pred))
print(confusion_matrix(y_test, y_pred)) '''



# ********************************** KNN Regressor ********************************
from sklearn.datasets import make_regression
X, y = make_regression(n_samples=1000, n_features=2, noise=10, random_state=42)
# sns.scatterplot(x= pd.DataFrame(X), y= pd.DataFrame(X)[1], hue=y)
# plt.show()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
from sklearn.neighbors import KNeighborsRegressor
regressor = KNeighborsRegressor(n_neighbors=6, algorithm='auto')
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

from sklearn.metrics import r2_score,confusion_matrix,classification_report, mean_absolute_error, mean_squared_error
print(r2_score(y_test, y_pred))
print(mean_absolute_error(y_test,y_pred))
print(mean_squared_error(y_test, y_pred))