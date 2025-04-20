import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('cardekho_imputated.csv', index_col=[0])
# print(df.head().to_string())

""" Data Cleaning
1. Handling missing values
2. Handling duplicates
3. Check data types
4. Understanding the data"""

# print(df.isnull().sum())
df.drop(columns=['car_name','brand'], inplace=True, axis=1)
# print(df.head().to_string())
# df['model'].unique()


# Getting all different types of feature
num_f = [features for features in df.columns if df[features].dtype != 'O']
cat_f = [features for features in df.columns if df[features].dtype == 'O']
discrete_f = [features for features in num_f if len(df[features].unique())<=25 ]
continuous_f = [features for features in num_f if features not in discrete_f ]
# print("Numerical Feature:", num_f)
# print("Categorical Feature:", cat_f)
# print("Discrete Feature:", discrete_f)
# print("Continuous Feature:", continuous_f)


# Independent and dependent data
X = df.drop(['selling_price'], axis=1)
y = df['selling_price']

# Feature encoding and Scaling
""" 
One Hot Encoding for Columns which had lesser unique values and not ordinal
One hot encoding is a process by which categorical variables are converted into a form 
that could be provided to ML algorithms to do a better job in prediction."""


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X['model'] = le.fit_transform(X['model'])

# Create a column Transformers with 3 types of transformers

num_f = X.select_dtypes(exclude='object').columns
onehot_columns = ['seller_type','fuel_type','transmission_type']

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

num_transformer = StandardScaler()
oh_transformer = OneHotEncoder(drop='first')

preprocessor = ColumnTransformer(
    [
    ('StandardScaler', num_transformer, num_f),
    ('OneHotEncoder', oh_transformer, onehot_columns)
    ], remainder = 'passthrough'
)

X = preprocessor.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# model training and model selection
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from  sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

# Create a function to evaluate model
def evaluate_model(true, predicted):
    mae = mean_absolute_error(true, predicted)
    mse = mean_squared_error(true, predicted)
    rmse = np.sqrt(mse)
    r2_s = r2_score(true, predicted)
    return mae, mse, rmse, r2_s


"""models = {
    'LinearRegression': LinearRegression(),
    'AdaboostRegression': AdaBoostRegressor(),
    'Ridge': Ridge(),
    'Lasso': Lasso(),
    'K_Neighbour Regressor': KNeighborsRegressor(),
    "Decision Tree":DecisionTreeRegressor(),
    'Random Forest Regressor': RandomForestRegressor(),
    'GradientBoost Regressor': GradientBoostingRegressor(),
    'XG boost Regressor': XGBRegressor()
}

for i in range(len(list(models))):
    model = list(models.values())[i]
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict((X_test))

    # Evaluate train and test data
    model_train_mae, model_train_mse, model_train_rmse, model_train_r2 = evaluate_model(y_train, y_train_pred)
    model_test_mae, model_test_mse, model_test_rmse, model_test_r2 = evaluate_model(y_test, y_test_pred)

    print(list(models.keys())[i])
    print('Model Performance for Training Set')
    print('mae:{:0.4f}'.format(model_train_mae))
    print('mse:{:0.4f}'.format(model_train_mse))
    print('rmse:{:0.4f}'.format(model_train_rmse))
    print('r2_Score:{:0.4f}'.format(model_train_r2))
    print('**************************************')
    print('Model Performance for Testing Set')
    print('mae:{:0.4f}'.format(model_test_mae))
    print('mse:{:0.4f}'.format(model_test_mse))
    print('rmse:{:0.4f}'.format(model_test_rmse))
    print('r2_Score:{:0.4f}'.format(model_test_r2))
    print('='*50)
    print('\n\n\n')"""


# Hyperparameter tuning
# knn = {'n_neighbors':[2,3,10,20,40,50]}
"""rf = {
    'max_depth': [5,8,15,None,10],
    'max_features': [5,7,'auto',8],
    'min_samples_split':[2,8,15,20],
    'n_estimators':[100,200,500,1000]
}
xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8, 12, 20, 30],
                  "n_estimators": [100, 200, 300],
                  "colsample_bytree": [0.5, 0.8, 1, 0.3, 0.4]}

randomcv_models = [
    # ('KNN', KNeighborsRegressor(), knn),
    ('RF', RandomForestRegressor(), rf),
    ('XGB', XGBRegressor(), xgboost_params)
]

from sklearn.model_selection import RandomizedSearchCV
model_param = {}
for name, model, param in randomcv_models:
    random = RandomizedSearchCV(estimator=model,
                                param_distributions=param,
                                n_iter=100,
                                cv=3,
                                verbose=2,
                                n_jobs=-1)
    random.fit(X_train, y_train)
    model_param[name] = random.best_params_

for model_name in model_param:
    print(f'*********best param for model {model_name}*******')
    print(model_param[model_name])"""



models = {
    "Random Forest Regressor": RandomForestRegressor(n_estimators=100, min_samples_split=2, max_features=7,
                                                     max_depth=15, n_jobs=-1),
    "XG Boost Regressor": XGBRegressor(n_estimators= 300,learning_rate=0.1,
                                     max_depth=5,colsample_bytree=0.8)

}
for i in range(len(list(models))):
    model = list(models.values())[i]
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict((X_test))

    # Evaluate train and test data
    model_train_mae, model_train_mse, model_train_rmse, model_train_r2 = evaluate_model(y_train, y_train_pred)
    model_test_mae, model_test_mse, model_test_rmse, model_test_r2 = evaluate_model(y_test, y_test_pred)

    print(list(models.keys())[i])
    print('Model Performance for Training Set')
    print('mae:{:0.4f}'.format(model_train_mae))
    print('mse:{:0.4f}'.format(model_train_mse))
    print('rmse:{:0.4f}'.format(model_train_rmse))
    print('r2_Score:{:0.4f}'.format(model_train_r2))
    print('**************************************')
    print('Model Performance for Testing Set')
    print('mae:{:0.4f}'.format(model_test_mae))
    print('mse:{:0.4f}'.format(model_test_mse))
    print('rmse:{:0.4f}'.format(model_test_rmse))
    print('r2_Score:{:0.4f}'.format(model_test_r2))
    print('=' * 50)
    print('\n\n\n')
