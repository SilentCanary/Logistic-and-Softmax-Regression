# -*- coding: utf-8 -*-
"""
Created on Fri Jun  6 01:10:37 2025

@author: advit
"""

from sklearn.datasets import load_digits
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,StandardScaler

wine=load_digits()
x=pd.DataFrame(wine.data)
y=pd.Series(wine.target)

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

encoder=OneHotEncoder(sparse_output=False,categories='auto')
y_onehot=encoder.fit_transform(y.values.reshape(-1,1))

x_train,x_test,y_train,y_test=train_test_split(x_scaled,y_onehot,test_size=0.2,random_state=42)

train_data=pd.DataFrame(data=pd.concat([pd.DataFrame(y_train),pd.DataFrame(x_train)],axis=1))
test_data=pd.DataFrame(data=pd.concat([pd.DataFrame(y_test),pd.DataFrame(x_test)],axis=1))

train_data.to_csv('train3.csv',index=False,header=False)
test_data.to_csv('test3.csv',index=False,header=False)
