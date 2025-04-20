from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report
X, y = make_classification(n_samples= 1000, n_features=10, n_informative=3, n_classes=3, random_state=15)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

from sklearn.linear_model import LogisticRegression
logistic = LogisticRegression(multi_class='ovr')
logistic.fit(X_train, y_train)
y_pred = logistic.predict(X_test)

score = accuracy_score(y_test, y_pred)
print(score)

confu_matrix = confusion_matrix(y_test, y_pred)
print(confu_matrix)

class_report = classification_report(y_test, y_pred)
print(class_report)