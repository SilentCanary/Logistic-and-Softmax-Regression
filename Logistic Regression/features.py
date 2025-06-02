# -*- coding: utf-8 -*-
"""
Created on Mon Jun  2 18:27:22 2025

@author: advit
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer

data=load_breast_cancer()

x=pd.DataFrame(data.data,columns=data.feature_names)
y=pd.Series(data.target,name='label')

df=pd.concat([x,y],axis=1)

corr=df.corr()

plt.figure(figsize=(18, 15))


sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", square=True, cbar=True, linewidths=0.5)
plt.show()

label_correlation = corr['label'].sort_values(ascending=False)

print("Correlation with target label:\n")
print(label_correlation)
