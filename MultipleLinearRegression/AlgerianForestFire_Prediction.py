""" EDA PART
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv("Algerian_forest_fires_cleaned_dataset.csv")
# print(dataset.head())
# print(dataset.info())

# print(dataset[dataset.isnull().any(axis=1)])

# EDA for dataset
# print(dataset)
df_copy = dataset.drop(['day','month', 'year'], axis=1)
# print(df_copy.head())

# print(df_copy['Classes'].value_counts())
df_copy['Classes'] = np.where(df_copy['Classes'].str.contains('not fire'),0,1)
# print(df_copy['Classes'].value_counts())

# EDA on dataset
# plot density plot for all feature
# plt.style.use("seaborn-v0_8-whitegrid")
# df_copy.hist(bins=100,figsize=(30,20))
# plt.show()

# Percentage for pie chart
percentage = df_copy['Classes'].value_counts(normalize=True)*100

# Plotting pie chart
classlabels = ["Fire", "Not Fire"]
# plt.figure(figsize=(12,7))
# plt.pie(percentage,labels=classlabels, autopct="%1.1f%%")
# plt.title("Pie Chart of classes")
# plt.show()

# Correlation of data
df_copy.corr()
# plt.figure()
# sns.heatmap(df_copy,annot=True)
# plt.show()

# sns.boxplot(df_copy['FWI'],color='green')
# plt.show()

EDA part ends here
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Algerian_forest_fires_cleaned_dataset.csv')
# print(df.head())

# removing day month and year as it will not have major role in output
df.drop(['day','month','year'],axis=1, inplace=True)
# print(df.head())

# converting object type data to integer
df['Classes'] = np.where(df['Classes'].str.contains('not fire'),0,1)
# print(df.head(), df.tail())

# Classifying independent and dependent feature
X= df.drop('FWI',axis=1)
y= df['FWI']

# print(X.head())
# print(y)

# Train test split of Data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=42)
# print(X_train.shape,X_test.shape)

# Feature selection based on correlation
# print(X_train.corr())

# Check for multi-collinearity
# plt.figure(figsize=(12,10))
# corr = X_train.corr()
# sns.heatmap(corr,annot=True)
# plt.show()


# removing highly correlated features
def correlation(dataset, threshold):
    col_corr = set()
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i][j]> threshold):
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr


corr_feature = correlation(X_train,0.85)
# print(corr_feature)

#dropping the correlated features from train and test data

X_train.drop(corr_feature,axis=1, inplace=True)
X_test.drop(corr_feature,axis=1, inplace=True)
# print(X_train.shape, X_test.shape)

# Feature scaling or Standardization
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Plotting box plot to understand the effect of Standard scaler
# plt.subplots(figsize=(15,5))
# plt.subplot(1,2,1)
# sns.boxplot(data=X_train)
# plt.title("X_train before scaling")
# plt.subplot(1,2,2)
# sns.boxplot(data=X_train_scaled)
# plt.title("X_train after scaling")
# plt.show()


# Applying Linear Regression model
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_absolute_error, r2_score
# linreg = LinearRegression()
# linreg.fit(X_train_scaled,y_train)
# y_pred = linreg.predict(X_test_scaled)
# mae = mean_absolute_error(y_test, y_pred)
# score = r2_score(y_test,y_pred)
# print("Mean Abs Error:", mae)
# print("R2 score:", score)
# plt.scatter(y_test,y_pred)
# plt.show()


# Applying Lasso Regression
# from sklearn.linear_model import Lasso
# from sklearn.metrics import mean_absolute_error, r2_score
# lassoreg = Lasso()
# lassoreg.fit(X_train_scaled, y_train)
# y_pred = lassoreg.predict(X_test_scaled)
# mae = mean_absolute_error(y_test, y_pred)
# score = r2_score(y_test,y_pred)
# print("Mean Abs Error:", mae)
# print("R2 score:", score)
# plt.scatter(y_test,y_pred)
# plt.show()


# Cross Validation with Lasso
# from sklearn.linear_model import LassoCV
# from sklearn.metrics import mean_absolute_error, r2_score
# lassocvreg = LassoCV(cv=5)
# lassocvreg.fit(X_train_scaled, y_train)
# y_pred = lassocvreg.predict(X_test_scaled)
# mae = mean_absolute_error(y_test, y_pred)
# score = r2_score(y_test,y_pred)
# print("Mean Abs Error:", mae)
# print("R2 score:", score)
# plt.scatter(y_test,y_pred)
# plt.show()

# Applying Ridge regression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
ridgereg = Ridge()
ridgereg.fit(X_train_scaled, y_train)
y_pred = ridgereg.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
score = r2_score(y_test,y_pred)
print("Mean Abs Error:", mae)
print("R2 score:", score)
plt.scatter(y_test,y_pred)
plt.show()


# Applying Ridge Cross Validation
# from sklearn.linear_model import RidgeCV
# from sklearn.metrics import mean_absolute_error, r2_score
# ridgecvreg = RidgeCV(cv=5)
# ridgecvreg.fit(X_train_scaled, y_train)
# y_pred = ridgecvreg.predict(X_test_scaled)
# mae = mean_absolute_error(y_test, y_pred)
# score = r2_score(y_test,y_pred)
# print("Mean Abs Error:", mae)
# print("R2 score:", score)
# plt.scatter(y_test,y_pred)
# plt.show()


# Applying Elasticnet
# from sklearn.linear_model import ElasticNet
# from sklearn.metrics import mean_absolute_error, r2_score
# elasticreg = ElasticNet()
# elasticreg.fit(X_train_scaled, y_train)
# y_pred = elasticreg.predict(X_test_scaled)
# mae = mean_absolute_error(y_test, y_pred)
# score = r2_score(y_test,y_pred)
# print("Mean Abs Error:", mae)
# print("R2 score:", score)
# plt.scatter(y_test,y_pred)
# plt.show()

# Appplying Elasticnet CV
# from sklearn.linear_model import ElasticNetCV
# from sklearn.metrics import mean_absolute_error, r2_score
# elasticcvreg = ElasticNetCV(cv=5)
# elasticcvreg.fit(X_train_scaled, y_train)
# y_pred = elasticcvreg.predict(X_test_scaled)
# mae = mean_absolute_error(y_test, y_pred)
# score = r2_score(y_test,y_pred)
# print("Mean Abs Error:", mae)
# print("R2 score:", score)
# plt.scatter(y_test,y_pred)
# plt.show()
# print(elasticcvreg.get_params())


# Pickling the file for deployment(Machine learning model and preprocessing model standardscaler)
# import pickle
# pickle.dump(scaler, open("scaler.pkl", 'wb'))
# pickle.dump(ridgereg, open("ridge.pkl", 'wb'))




