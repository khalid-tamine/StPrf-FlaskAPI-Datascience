#!/usr/bin/env python
# coding: utf-8
"""
Created on Tue otocber 13 12:50:56 2020

@author: khalid-tamine
"""
# In[6]:


import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
import seaborn as sns
import pandas as pd
import numpy as np 
from sklearn.metrics import mean_absolute_error

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import cross_validate

from sklearn.model_selection import cross_val_score
from math import sqrt
from sklearn.model_selection import train_test_split

df = pd.read_csv('C:/Users/LOMEN/Desktop/student/student-mat.csv',delimiter =";")



# In[28]:

df.head()



# In[19]:


df.describe()


# In[20]:


df.columns


# In[21]:


df.G1.hist()


# In[22]:


df.G2.hist()


# In[23]:


df.boxplot(column =['G1','G2','G3'])


# In[24]:


df[['romantic','health','Walc','absences','freetime','activities','famrel','traveltime','internet','Dalc','famsup','G1','G2','G3']].corr()


# In[25]:


cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(df[['romantic','health','Walc','absences','freetime','activities','famrel','traveltime','internet','Dalc','famsup','G1','G2','G3']].corr(),vmax=.3, center=0, cmap=cmap,
             square=True, linewidths=.5, cbar_kws={"shrink": .5})


# In[7]:


features = df[['romantic','health','Walc','absences','freetime','activities','famrel','traveltime','internet','Dalc','famsup','G1','G2','G3']]
df_dm = pd.get_dummies(features)

X = df_dm.drop('G3' , axis = 1)
y = df_dm.G3.values
train_X, val_X, train_y, val_y = train_test_split(X, y, test_size=0.2,random_state=200)

#RandomForest model
iowa_model = RandomForestRegressor(n_estimators=110,random_state=1)
iowa_model.fit(train_X, train_y)

np.mean(cross_val_score(iowa_model,train_X, train_y,scoring = 'neg_mean_absolute_error', cv= 3))


# In[8]:

#linear regression model
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.model_selection import cross_val_score

lm = LinearRegression()
lm.fit(train_X, train_y)

np.mean(cross_val_score(lm,train_X,train_y, scoring = 'neg_mean_absolute_error', cv= 3))


# In[9]:

#lasso model
lm_l = Lasso(alpha=.13)
lm_l.fit(train_X, train_y)
np.mean(cross_val_score(lm_l,train_X, train_y, scoring = 'neg_mean_absolute_error', cv= 3))


# In[10]:

#GridSearch to improve randomForest model
from sklearn.model_selection import GridSearchCV
parameters = {'n_estimators':range(10,300,10), 'criterion':('mse','mae'), 'max_features':('auto','sqrt','log2')}

gs = GridSearchCV(iowa_model,parameters,scoring='neg_mean_absolute_error',cv=3)
gs.fit(train_X, train_y)

gs.best_score_
gs.best_estimator_


# In[81]:


#test predictions
tpred_lm = lm.predict(val_X)
tpred_lml = lm_l.predict(val_X)
tpred_rf = gs.best_estimator_.predict(val_X)

#sees result of predictions accuracy
mean_absolute_error(val_y,tpred_lm)
mean_absolute_error(val_y,tpred_lml)
mean_absolute_error(val_y,tpred_rf)


# In[91]:

#dividing prediction of sum of two models by 2 to test if I get a better result
mean_absolute_error(val_y,(tpred_lml+tpred_rf)/2)


# In[96]:


plt.plot(val_y-tpred_rf,marker='o',linestyle='')


# In[11]:


import pickle
pickl = {'model': gs.best_estimator_}
pickle.dump( pickl, open( 'model_file' + ".p", "wb" ) )

file_name = "model_file.p"
with open(file_name, 'rb') as pickled:
    data = pickle.load(pickled)
    model = data['model']

model.predict(np.array(list(val_X.iloc[1,:])).reshape(1,-1))[0]

list(val_X.iloc[1,:])


# In[12]:


list(val_X.iloc[1,:])


# In[ ]:




