import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

data = pd.read_csv("Mlnotebooks\data\ObesityDataSet_raw_and_data_sinthetic.csv", index_col=[0])

nominal = ['Gender','family_history_with_overweight','FAVC','CAEC','SMOKE','SCC','CALC','MTRANS']

numerical = ['TUE','FAF','CH2O','NCP','FCVC','Weight','Height','Age']

logreg = LogisticRegression()
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()
svm = SVC()
scaler = StandardScaler()

encoded = pd.get_dummies(data[nominal])
data = pd.concat([data.drop(nominal, axis=1), encoded], axis=1)

data[numerical] = scaler.fit_transform(data[numerical])

X = data.drop('NObeyesdad', axis=1)
y = data['NObeyesdad']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

logreg.fit(X_train, y_train)
dt.fit(X_train, y_train)
rf.fit(X_train, y_train)
svm.fit(X_train, y_train)

logreg_preds = logreg.predict(X_test)
dt_preds = dt.predict(X_test)
rf_preds = rf.predict(X_test)
svm_preds = svm.predict(X_test)

# Step 7: Model Evaluation
# Evaluate the performance of each model
logreg_accuracy = accuracy_score(y_test, logreg_preds)
dt_accuracy = accuracy_score(y_test, dt_preds)
rf_accuracy = accuracy_score(y_test, rf_preds)
svm_accuracy = accuracy_score(y_test, svm_preds)

print("Logistic Regression Accuracy:", logreg_accuracy)
print("Decision Tree Accuracy:", dt_accuracy)
print("Random Forest Accuracy:", rf_accuracy)
print("Support Vector Machine Accuracy:", svm_accuracy)

# Additional Evaluation Metrics
print("Logistic Regression Report:")
print(classification_report(y_test, logreg_preds))

print("Decision Tree Report:")
print(classification_report(y_test, dt_preds))

print("Random Forest Report:")
print(classification_report(y_test, rf_preds))

print("Support Vector Machine Report:")
print(classification_report(y_test, svm_preds))