import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import numpy as np

data = pd.read_csv('Mlnotebooks\data\online_shoppers_intention.csv')

nominal = ['Month','VisitorType']
encoded = pd.get_dummies(data[nominal])


data = pd.concat([data.drop(nominal, axis=1), encoded], axis=1)

X = data.drop(['Revenue', 'Weekend', 'Informational_Duration'], axis=1)
y_revenue = data['Revenue']
y_weekend = data['Weekend']
y_duration = data['Informational_Duration']
X_train, X_test, y_revenue_train, y_revenue_test, y_weekend_train, y_weekend_test, y_duration_train, y_duration_test = train_test_split(X, y_revenue, y_weekend, y_duration, test_size=0.2, random_state=42)

weekend_model = RandomForestClassifier(n_estimators=100)

weekend_model.fit(X_train, y_weekend_train)

weekend_predictions = weekend_model.predict(X_test)

revenue_model = RandomForestClassifier(n_estimators=100)

revenue_model.fit(X_train, y_revenue_train)

revenue_predictions = revenue_model.predict(X_test)

duration_model = RandomForestRegressor(n_estimators=100)

duration_model.fit(X_train, y_duration_train)

duration_predictions = duration_model.predict(X_test)

# Test revenue predictions
revenue_accuracy = np.mean(revenue_predictions == y_revenue_test)
print("Revenue Accuracy:", revenue_accuracy)

# Test weekend predictions
weekend_accuracy = np.mean(weekend_predictions == y_weekend_test)
print("Weekend Accuracy:", weekend_accuracy)

# Test duration predictions
duration_mse = np.mean((duration_predictions - y_duration_test) ** 2)
print("Duration MSE:", duration_mse)