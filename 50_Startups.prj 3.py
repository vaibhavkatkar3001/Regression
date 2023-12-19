#!/usr/bin/env python
# coding: utf-8

# In[1]:


import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


data = pd.read_csv('C:/Users/soura/OneDrive/Desktop/Vaibhav DS/89/MLR project/50_Startups.csv')
data.head()


# In[3]:


x = data.drop(['Profit','State'], axis=1).values
y = data['Profit'].values


# In[4]:


print(x)


# In[5]:


print(y)


# In[6]:


data.shape


# In[7]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)


# In[8]:


x_train


# In[8]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train,y_train)


# In[9]:


y_pred = model.predict(x_test)
print(y_pred)


# In[10]:


model.predict([[165349.20,136897.80,471784.10]])


# In[11]:


from sklearn.metrics import r2_score
r2_score(y_test,y_pred)


# In[15]:


plt.figure(figsize=(10,5))
plt.scatter(y_test,y_pred)


# In[ ]:




