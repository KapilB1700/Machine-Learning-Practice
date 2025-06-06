import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')
df = pd.read_csv('Travel.csv')
# print(df.head().to_string())


# Data Cleaning
'''
1. Handling Missing values
2. Handling Duplicates
3. Check data type
4. Understand the data'''

# Check missing value
# print(df.isnull().sum())


# Handling duplicates
# print(df['Gender'].value_counts())
# print(df['MaritalStatus'].value_counts())
# print(df['TypeofContact'].value_counts())

df['Gender'] = df['Gender'].replace('Fe Male', 'Female')
df['MaritalStatus'] = df['MaritalStatus'].replace('Single', 'Unmarried')

# print(df.head().to_string())
# print(df['Gender'].value_counts())
# print(df['MaritalStatus'].value_counts())

# Check missing values
# feature with nan values
features_with_na = [features for features in df.columns if df[features].isnull().sum()>=1]
# for feature in features_with_na:
    # print(feature,np.round(df[feature].isnull().mean()*100,5), '% missing values')

# statistics for  numerical columns (Null Cols)
# print(df[features_with_na].select_dtypes(exclude='object').describe().to_string())

'''For imputing null values
1. for numerical columns median will be used
2. for categorical columns mode will be used'''

df.Age.fillna(df.Age.median(), inplace=True)
df.TypeofContact.fillna(df.TypeofContact.mode()[0], inplace=True)
df.DurationOfPitch.fillna(df.DurationOfPitch.median(), inplace=True)
df.NumberOfFollowups.fillna(df.NumberOfFollowups.mode()[0], inplace=True)
df.PreferredPropertyStar.fillna(df.PreferredPropertyStar.mode()[0], inplace=True)
df.NumberOfTrips.fillna(df.NumberOfTrips.median(), inplace=True)
df.NumberOfChildrenVisiting.fillna(df.NumberOfChildrenVisiting.median(), inplace=True)
df.MonthlyIncome.fillna(df.MonthlyIncome.median(), inplace=True)

# print(df.isnull().sum())
df.drop('CustomerID', inplace=True, axis=1)

# Feature Engineering
# Merge two columns people visiting and nu of children visiting
df['TotalVisiting'] = df['NumberOfPersonVisiting'] + df['NumberOfChildrenVisiting']
df.drop(columns = ['NumberOfPersonVisiting', 'NumberOfChildrenVisiting'], axis=1, inplace=True)


# fetching all the numeric features
num_features = [features for features in df.columns if df[features].dtype != 'O']
# print('Number of numerical feature', len(num_features))

# fetching all the categorical features
cat_features = [features for features in df.columns if df[features].dtype == 'O']
# print('Number of numerical feature', len(cat_features))

# fetching all the discrete features
discrete_features = [features for features in num_features if len(df[features].unique()) <=25]
# print('Number of discrete feature', len(discrete_features))

# fetching all the continuous features
continuous_features = [features for features in num_features if features not in discrete_features]
# print('Number of continuous feature', len(continuous_features))


# Feature engineering for Categorical features
from sklearn.model_selection import train_test_split
X = df.drop(['ProdTaken'], axis=1)
y = df['ProdTaken']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
cat_features = X.select_dtypes(include='object').columns
num_features = X.select_dtypes(exclude='object').columns

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
numeric_transformer = StandardScaler()
oh_transformer = OneHotEncoder(drop='first')

preprocessor = ColumnTransformer(
    [
        ('OneHotEncoder', oh_transformer, cat_features),
        ('StandardScaler', numeric_transformer, num_features)
    ]
)

X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform((X_test))


# ML model training
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,ConfusionMatrixDisplay,
                             roc_auc_score ,roc_curve, f1_score,recall_score,precision_score)

models = {
   "Random Forest": RandomForestClassifier(),
    # "Decision Tree": DecisionTreeClassifier(),
    # "Logistic Regression": LogisticRegression()
}

"""for i in range(len(list(models))):
    model = list(models.values())[i]
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Training set Performance
    model_train_accuracy = accuracy_score(y_train, y_train_pred, )
    model_train_f1 = f1_score(y_train, y_train_pred, average='weighted')
    model_train_precision = precision_score(y_train, y_train_pred, )
    model_train_recall = recall_score(y_train, y_train_pred, )
    model_train_rocauc_score = roc_auc_score(y_train, y_train_pred)

    # Test set Performance
    model_test_accuracy = accuracy_score(y_test, y_test_pred)
    model_test_f1 = f1_score(y_test , y_test_pred, average='weighted')
    model_test_precision = precision_score(y_test, y_test_pred)
    model_test_recall = recall_score(y_test, y_test_pred)
    model_test_rocauc_score = roc_auc_score(y_test, y_test_pred)

    print(list(models.keys())[i])
    print('Training Data')
    print('Accuracy Score:{:.4f}'.format(model_train_accuracy))
    print('f1 Score:{:.4f}'.format(model_train_f1))
    print('Precision Score:{:.4f}'.format(model_train_precision))
    print('Recall Score:{:.4f}'.format(model_train_recall))
    print('ROC AUC Score:{:.4f}'.format(model_train_rocauc_score))
    print('-'*30)

    print('Testing Data')
    print('Accuracy Score:{:.4f}'.format(model_test_accuracy))
    print('f1 Score:{:.4f}'.format(model_test_f1))
    print('Precision Score:{:.4f}'.format(model_test_precision))
    print('Recall Score:{:.4f}'.format(model_test_recall))
    print('ROC AUC Score:{:.4f}'.format(model_test_rocauc_score))
    print('='*50)
    print('\n\n\n')
"""

# Hyperparameter Tuning
param = {
    'max_depth':[5,8,15,None,10],
    'max_features': [5,7,'auto', 8],
    'min_samples_split': [2,8,15,20],
    'n_estimators': [100,200,500,1000]

}

randomcv_models = [
    ('RF', RandomForestClassifier(), param),
]

from sklearn.model_selection import RandomizedSearchCV
"""model_param = {}
for name, model, params in randomcv_models:
    random = RandomizedSearchCV(estimator=model,
                                param_distributions=params,
                                n_iter=100,
                                cv=3,
                                verbose=2,
                                n_jobs=1)
    random.fit(X_train, y_train)
    model_param[name] = random.best_params_

for model_name in model_param:
    print(f"-------best params for {model_name}-----------")
    print(model_param[model_name])"""

models = {
   "Random Forest": RandomForestClassifier(n_estimators= 500, min_samples_split= 2,
                                           max_features= 8, max_depth=None)
}

for i in range(len(list(models))):
    model = list(models.values())[i]
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Training set Performance
    model_train_accuracy = accuracy_score(y_train, y_train_pred, )
    model_train_f1 = f1_score(y_train, y_train_pred, average='weighted')
    model_train_precision = precision_score(y_train, y_train_pred, )
    model_train_recall = recall_score(y_train, y_train_pred, )
    model_train_rocauc_score = roc_auc_score(y_train, y_train_pred)

    # Test set Performance
    model_test_accuracy = accuracy_score(y_test, y_test_pred)
    model_test_f1 = f1_score(y_test , y_test_pred, average='weighted')
    model_test_precision = precision_score(y_test, y_test_pred)
    model_test_recall = recall_score(y_test, y_test_pred)
    model_test_rocauc_score = roc_auc_score(y_test, y_test_pred)

    print(list(models.keys())[i])
    print('Training Data')
    print('Accuracy Score:{:.4f}'.format(model_train_accuracy))
    print('f1 Score:{:.4f}'.format(model_train_f1))
    print('Precision Score:{:.4f}'.format(model_train_precision))
    print('Recall Score:{:.4f}'.format(model_train_recall))
    print('ROC AUC Score:{:.4f}'.format(model_train_rocauc_score))
    print('-'*30)

    print('Testing Data')
    print('Accuracy Score:{:.4f}'.format(model_test_accuracy))
    print('f1 Score:{:.4f}'.format(model_test_f1))
    print('Precision Score:{:.4f}'.format(model_test_precision))
    print('Recall Score:{:.4f}'.format(model_test_recall))
    print('ROC AUC Score:{:.4f}'.format(model_test_rocauc_score))
    print('='*50)
    print('\n\n\n')

plt.figure()
auc_models = [
    {
        'label': "Random Forest Classifier",
        'model': RandomForestClassifier(n_estimators= 500, min_samples_split= 2,
                                           max_features= 8, max_depth=None),
        'auc': 0.8450
    },
]

# Create loop through all model
for algo in auc_models:
    model = algo['model']  # select the model
    model.fit(X_train, y_train)
    # Compute False positive rate, and True positive rate
    fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
    # Calculate Area under the curve to display on the plot
    plt.plot(fpr, tpr, label='%s ROC (area = %0.2f)' % (algo['label'], algo['auc']))
# Custom settings for the plot
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('1-Specificity(False Positive Rate)')
plt.ylabel('Sensitivity(True Positive Rate)')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.savefig("auc.png")
plt.show()