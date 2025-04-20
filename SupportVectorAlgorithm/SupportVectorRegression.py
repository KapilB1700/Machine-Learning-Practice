import seaborn as sns
df = sns.load_dataset('tips')
# print(df.head())
# print(df.info())
# print(df.columns)
# print(df['day'].value_counts())
# print(df['sex'].value_counts())
# print(df['smoker'].value_counts())
# print(df['time'].value_counts())

# independent and dependent feature
X = df[['tip', 'sex', 'smoker', 'day', 'time', 'size']]
y = df['total_bill']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10)

# Sex, Time, Smoker are binary categories(having only two values)
# so, we can perform feature encoding here(Label encoding:binary values
# and Onehot encoding- for multiple values)

from sklearn.preprocessing import LabelEncoder
le1 = LabelEncoder()
le2 = LabelEncoder()
le3 = LabelEncoder()
X_train['sex'] = le1.fit_transform(X_train['sex'])
X_train['smoker'] = le2.fit_transform(X_train['smoker'])
X_train['time'] = le3.fit_transform(X_train['time'])


X_test['sex'] = le1.transform(X_test['sex'])
X_test['smoker'] = le2.transform(X_test['smoker'])
X_test['time'] = le3.transform(X_test['time'])

# Onehot Encoding
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(transformers=[('onehot', OneHotEncoder(drop='first'),[3])], remainder='drop')
# where 3 is third index that is index of column

X_train = ct.fit_transform(X_train)
X_test = ct.transform(X_test)

# to show complete output or data format
import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)

# support vector regressor
from sklearn.svm import SVR
svr = SVR()

svr.fit(X_train, y_train)
y_pred = svr.predict(X_test)

from sklearn.metrics import r2_score,mean_absolute_error
print(r2_score(y_test, y_pred))
print(mean_absolute_error(y_test, y_pred))

# Hyperparameter tuning using gridsearchCV

from sklearn.model_selection import GridSearchCV
param = {
    'C':[0.1,1,10,100,1000],
    'gamma':[1,0.1,0.01,0.001,0.0001],
    'kernel': ['rbf', 'sigmoid', 'linear', 'poly']
}

grid = GridSearchCV(estimator=SVR(), param_grid=param, cv=5, refit=True, verbose=3)
grid.fit(X_train, y_train)
y_pred = grid.predict(X_test)

print('****************************************************************')
print(r2_score(y_test, y_pred))
print(mean_absolute_error(y_test, y_pred))
print(grid.best_score_)
print(grid.best_params_)

