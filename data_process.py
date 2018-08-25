import sys
import re
import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split

def get_title(name):
    title_search = re.search(r'([A-Za-z]+)\.', name)
    #如果称呼存在，返回称呼
    if title_search:
        return title_search.group(1)
    return ""

def preprocess_dataset(datasets):
    for dataset in datasets:
        #性别映射为数值，0,1
        dataset['Sex']=dataset['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

        # 将SibSp和Parch两个合并为一个特征，家庭大小，并同时扩展为是否独自一人的特征
        dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
        dataset['IsAlone'] = 0
        dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

        # 称呼分别为0，1，2，3，4，5，5为没有称呼
        dataset['Title'] = dataset['Name'].apply(get_title)
        dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
            'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
        dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
        dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
        title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
        dataset['Title'] = dataset['Title'].map(title_mapping)
        dataset['Title'] = dataset['Title'].fillna(0)

        # 终点站，缺失值补充为S，有三种类型 
        dataset['Embarked'] = dataset['Embarked'].fillna('S')
        dataset['Embarked'] = dataset['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

        # 票价，0，1，2，3四种
        dataset['Fare'] = dataset['Fare'].fillna(dataset['Fare'].median())
        dataset.loc[dataset['Fare'] <= 7.91, 'Fare'] = 0
        dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
        dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare'] = 2
        dataset.loc[ dataset['Fare'] > 31, 'Fare'] = 3
        dataset['Fare'] = dataset['Fare'].astype(int)

        # 年龄，缺失值为类别5
        dataset.loc[ dataset['Age'] <= 16, 'Age'] = 0
        dataset.loc[(dataset['Age'] > 16) & (dataset['Age'] <= 32), 'Age'] = 1
        dataset.loc[(dataset['Age'] > 32) & (dataset['Age'] <= 48), 'Age'] = 2
        dataset.loc[(dataset['Age'] > 48) & (dataset['Age'] <= 64), 'Age'] = 3
        dataset.loc[ dataset['Age'] > 64, 'Age']= 4
        dataset.loc[ dataset['Age'].isnull(), 'Age']= 5
        dataset['Age'] = dataset['Age'].astype(int)
    
if __name__ == '__main__':
    train_data=pd.read_csv("./data/train.csv")
    test_data=pd.read_csv("./data/test.csv")
    datasets=[train_data,test_data]
    preprocess_dataset(datasets)
    drop_elements = ['PassengerId', 'Name', 'Ticket', 'Cabin', 'SibSp','Parch']
    train_data = train_data.drop(drop_elements, axis = 1)
    test_data  = test_data.drop(drop_elements, axis = 1)

    x_train,x_valid,y_train,y_valid=train_test_split(
        train_data.drop(['Survived'],axis=1),train_data['Survived'],test_size=0.2,random_state=24)
    print(x_train.head(5).as_matrix())
    print(y_train.head(5).as_matrix())





