import pandas as pd
from sklearn.datasets import load_diabetes
dataset = load_diabetes()
# print(dataset['DESCR'])

# splitting data into independent and dependent feature
X = pd.DataFrame(dataset.data, columns=['age','sex','bmi','bp','s1','s2','s3','s4','s5','s6'])
y = dataset['target']
# print(X.head())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=10)
# print(X_train.corr())

import seaborn as sns
import matplotlib.pyplot as plt
sns.heatmap(X_train.corr(), annot=True)
# plt.show()

from sklearn.tree import DecisionTreeRegressor
reg = DecisionTreeRegressor()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)

from sklearn.metrics import r2_score,mean_absolute_error, mean_squared_error
# print("r2_score:",r2_score(y_test, y_pred))
# print("mean_absolute_error:",mean_absolute_error(y_test, y_pred))
# print("mean_squared_error:",mean_squared_error(y_test, y_pred))

# Hyperparameter Tuning
params = {
    'criterion': ['squared_error','friedman_mse','absolute_error'],
    'splitter': ['random', 'best'],
    'max_depth': [1,2,3,4,5,10,15,20,25],
    'max_features': ['auto', 'sqrt', 'log2']
}
from sklearn.model_selection import GridSearchCV
grid = GridSearchCV(DecisionTreeRegressor(), param_grid=params, cv=5, scoring='neg_mean_squared_error')
import warnings
warnings.filterwarnings('ignore')

grid.fit(X_train, y_train)
# print('\nHyperparameter Tuning\n')
# print(grid.best_score_)
# print(grid.best_params_)
y_pred = grid.predict(X_test)
# print("r2_score:", r2_score(y_test, y_pred))
# print("mean_absolute_error:", mean_absolute_error(y_test, y_pred))
# print("mean_squared_error:", mean_squared_error(y_test, y_pred))


model = DecisionTreeRegressor(criterion='friedman_mse', max_depth=4, max_features='log2', splitter='random')
model.fit(X_train, y_train)
# print(model)

# Visualize tree model
import matplotlib.pyplot as plt
from sklearn import tree
tree.plot_tree(model, filled=True)
plt.show()