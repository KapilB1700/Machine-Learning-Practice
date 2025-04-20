import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the dataset
df = pd.read_csv("height-weight.csv")
# print(df.head())

# Plotting the relation between dataset
# plt.scatter(df["Weight"], df["Height"])
# plt.xlabel('Weight')
# plt.ylabel('Height')
# plt.show()

# Divide dataset into independent(X) and dependent dataset(Y)
X = df[['Weight']]
y = df['Height']
# print(X,y)

# Splitting the data for training and test
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=42)
# print(X.shape, y.shape)
# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

'''standardization of train data(independent feature) for
1. Faster calculation of  gradient descend(optimization) and
2. because both feature are in different unit '''

from sklearn.preprocessing import StandardScaler  # applying Z-score formula
scaler= StandardScaler()
# We need to standardize test data as well based on observations of trained data

X_train = scaler.fit_transform(X_train)
# This will calculate mean standard deviation and then apply z score
# fit will compute mean and standard deviation and transform will apply z score on every data set

'''But for test data we dont need to calculate mean and standard deviation
again. This will be taken from train data 
so we will apply just transform to calculate z score'''

X_test = scaler.transform(X_test)
# print(X_train, X_test)
# plt.scatter(X_train, y_train)
# plt.show()

# Train simple linear regression model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

# Training the model on data (fit-keyword)
regressor.fit(X_train, y_train)
# print(regressor.coef_)  # slope or coefficient of weight
# print(regressor.intercept_)  # intercept point (y = mx+ C)
# plt.scatter(X_train,y_train)
# plt.show()

# plt.scatter(X_train, regressor.predict(X_train))
# plt.show()
#
# plt.plot(X_train, regressor.predict(X_train))
# plt.show()

# Prediction of the output
y_pred_test = regressor.predict(X_test)
# print(y_test, y_pred_test) # Comparing with actual test data

''' Performance metrics
1. Calculating Mean Squared Error(MSE), Mean Absolute Error(MAE), Root MSE
   (Errors remaining in the model)
2. R Square and adjusted R square '''

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
mse = mean_squared_error(y_test, y_pred_test)  # difference check of actual and predicted value
mae = mean_absolute_error(y_test, y_pred_test)
rmse = np.sqrt(mse)
# print("MSE:", mse)
# print("MAE:", mae)
# print("RMSE:", rmse)

# R square value
score = r2_score(y_test, y_pred_test)
# print("R square (Score):", score)

# Adjusted R square
adj_score = 1 - (1-score)*(len(y_test)-1)/(len(y_test)-X_test.shape[1]-1)
# print("Adjusted R Square:", adj_score)

# Prediction of Height, Enter weight as input
Weight = int(input("Enter Weight:"))
scaled_weight = scaler.transform([[Weight]])  # Need to transform as
print("Height of person weighing", Weight, " will be:", regressor.predict([scaled_weight[0]]))


"""Three assumption for considering model as best
1. plotting y_test and y_pred_test should be close linear
2. Residual that is y_test-y_pred_test should follow normal distribution
3. Residual and y_pred_test plotting should be uniformly distributed (plus and minus values)"""



















