import pandas as pd 
import numpy as np 
import seaborn as sns 
import matplotlib.pyplot as plt
import gzip
import random
import warnings

n = 40428967  #total number of records in the clickstream data 
sample_size = 3000000
skip_values = sorted(random.sample(range(1,n), n-sample_size))

with gzip.open('../input/avazu-ctr-prediction/train.gz') as f:
    train = pd.read_csv(f,skiprows = skip_values)
train['hour'] = pd.to_datetime(train['hour'],format = '%y%m%d%H')

X = train.drop('click',axis=1)
y = train.click

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state = 42)

num_cols = X.select_dtypes(include = ['int','float']).columns.tolist()
categorical_cols = X.select_dtypes(include = ['object']).columns.tolist()
print(num_cols)
print(categorical_cols)


for col in categorical_cols:
	X_train[col] = X_train[col].apply(lambda x: hash(x))
    
for col in categorical_cols:
    X_test[col] = X_test[col].apply(lambda x:hash(x))

from sklearn.preprocessing import StandardScaler
std = StandardScaler()
X_train[num_cols] = std.fit_transform(X_train[num_cols])
X_test[num_cols] = std.transform(X_test[num_cols])

X_train['user_info'] = X_train.device_ip + X_train.device_model + X_train.device_id
X_train = X_train.drop(['device_id','device_ip','device_model','id','hour'],axis=1)
    
X_train['device_info'] = X_train.device_type + X_train.banner_pos + X_train.device_conn_type
X_train = X_train.drop(['banner_pos','device_conn_type','device_type'],axis=1)

X_test['user_info'] = X_test.device_ip + X_test.device_model + X_test.device_id
X_test = X_test.drop(['device_id','device_ip','device_model','id','hour'],axis=1)
    
X_test['device_info'] = X_test.device_type + X_test.banner_pos + X_test.device_conn_type
X_test = X_test.drop(['banner_pos','device_conn_type','device_type'],axis=1)

from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(max_depth = 10)
tree.fit(X_train,y_train)
print('Train Score:',tree.score(X_train,y_train))
print('Test Score:',tree.score(X_test,y_test))

from sklearn.metrics import roc_curve,confusion_matrix,precision_score,recall_score,roc_auc_score
y_score = tree.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_score[:, 1])
roc_auc_score = roc_auc_score(y_test,y_score[:,1])
print(roc_auc_score)