import pandas as pd 
import numpy as np 
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score,roc_auc_score
from data_process import preprocess_dataset

train_data=pd.read_csv("./data/train.csv")
test_data=pd.read_csv("./data/test.csv")
datasets=[train_data,test_data]
preprocess_dataset(datasets)
drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp','Parch']
train_data = train_data.drop(drop_elements, axis = 1)
test_data  = test_data.drop(drop_elements, axis = 1)

x_train,x_valid,y_train,y_valid=train_test_split(
        train_data.drop(['Survived'],axis=1),
        train_data['Survived'],test_size=0.2,random_state=24)

x_train=x_train.as_matrix()
y_train=y_train.as_matrix()
x_valid=x_valid.as_matrix()
y_valid=y_valid.as_matrix()


clf=XGBClassifier(
    booster='gbtree',nthread=4,
    learning_rate=0.1,min_child_weight=1,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=1,
    objective='binary:logistic',
    seed=24
    )
eval_set=[(x_valid,y_valid)]
clf.fit(x_train,y_train,early_stopping_rounds=100,
    eval_metric='logloss',eval_set=eval_set,verbose=True)
clf.save_model("./model/xgboost.model")
pre=clf.predict(x_valid)
pro=clf.predict_proba(x_valid)[:,1]

print("Accuracy: %f"%accuracy_score(y_valid,pre))
print("AUC Score: %f"%roc_auc_score(y_valid,pro))