# @Author: Yann Huet
# @Date:   2015-09-09T17:15:45+02:00
# @Email:  https://github.com/Alumet
# @Last modified by:   Yann Huet
# @Last modified time: 2016-05-09T16:38:02+02:00
# @License: MIT License (MIT), Copyright (c) Yann Huet

'''
    First experience with Random Forest Tree
    Kaggle Titanic dataset https://www.kaggle.com/c/titanic
    0.88 average score
'''

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score
import pandas as pd

#data importation form training data set
X=pd.read_csv("train-1.csv")
Y=X.pop("Survived")

#drop unique values
X.drop(["PassengerId"], axis=1,inplace=True)

#replace all Missing values by a new class called Missing
categorical_variables=["Sex","Cabin-letter","Name","Embarked","Ticket (prefix)"]

for variable in categorical_variables:
    X[variable].fillna("Missing",inplace=True)
    dummies=pd.get_dummies(X[variable], prefix=variable)
    X=pd.concat([X,dummies],axis=1)
    X.drop([variable],axis=1,inplace=True)

X.to_csv("X.csv",sep=';')

#creating new model with cleaned data set
model=RandomForestRegressor(1000, oob_score=True,n_jobs=-1, random_state=42, max_features="auto",min_samples_leaf=1)
model.fit(X,Y)
print ("C-stat:",roc_auc_score(Y,model.oob_prediction_))

#creat graph with features_importances
feature_importances=pd.Series(model.feature_importances_, index=X.columns)
feature_importances.sort_values(inplace=True)
feature_importances.plot(kind="barh",figsize=(7,14))
