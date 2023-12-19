#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


data = pd.DataFrame({'delivery time':[21,13.5,19.75,24,29,15.35,19,9.5,17.9,18.75,19.83,10.75,16.68,11.5,12.03,14.88,13.75,18.11,8,17.83,21.5],
                    'sorting time':[10,4,6,9,10,6,7,3,10,9,8,4,7,3,3,4,6,7,2,7,5]})
print(data)


# In[3]:


x = data['delivery time'].values.reshape(-1,1)
y = data['sorting time'].values.reshape(-1,1)


# In[4]:


sns.scatterplot(y = 'sorting time', x = 'delivery time', data = data)


# In[5]:


##### simple linear regression -----
from sklearn.linear_model import LinearRegression


# In[6]:


model = LinearRegression()


# In[7]:


model.fit(x,y)


# In[9]:


plt.figure(figsize=(5,3))
plt.scatter(x,y,color = 'blue')
plt.plot(x,model.predict(x), color = 'red')


# In[10]:


from sklearn.metrics import r2_score


# In[12]:


r2_score(y,model.predict(x))


# In[ ]:




