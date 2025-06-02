# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 23:25:12 2025

@author: advit
"""

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data=load_breast_cancer()
x=pd.DataFrame(data.data,columns=data.feature_names)
y=pd.Series(data.target,name='label')

selected_features=['mean concavity','mean area' ,'mean radius','mean perimeter','mean concave points'];
x_selected=x[selected_features]
x_train,x_test,y_train,y_test=train_test_split(x_selected, y,test_size=0.2,random_state=42)

scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train)
x_test_scaled=scaler.transform(x_test)

df_train=pd.DataFrame(x_train_scaled,columns=selected_features)
df_train['label']=y_train.reset_index(drop=True)

df_test=pd.DataFrame(x_test_scaled,columns=selected_features)
df_test['label']=y_test.reset_index(drop=True)

df_train.to_csv('train2.csv',index=False);
df_test.to_csv("test2.csv",index=False)
