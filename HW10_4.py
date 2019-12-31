#!/usr/bin/env python
# coding: utf-8

# In[13]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_iris


# In[14]:


iris = load_iris()


# In[15]:


x = iris.data
y = iris.target


# In[16]:


X = x[:,2:]


# In[33]:


#X


# In[18]:


Y = y


# In[19]:


len(x)


# In[20]:


from sklearn.model_selection import train_test_split


# In[21]:


x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)


# In[22]:


plt.scatter(X[:,0], X[:,1], c=Y, cmap='Paired')


# In[24]:


from sklearn.svm import SVC


# In[25]:


clf = SVC(gamma='auto')


# In[26]:


clf.fit(x_train, y_train)


# In[27]:


y_predict = clf.predict(x_test)


# In[28]:


y_predict


# In[29]:


y_test


# In[30]:


y_predict-y_test


# In[31]:


plt.scatter(x_test[:,0], x_test[:,1], c=y_predict-y_test)


# In[32]:


x0 = np.arange(0.5, 7, 0.02)
y0 = np.arange(0, 3, 0.02)

xm, ym = np.meshgrid(x0, y0)
P = np.c_[xm.ravel(), ym.ravel()]
z = clf.predict(P)

Z = z.reshape(xm.shape)
plt.contourf(xm, ym, Z, cmap='Paired', alpha=0.3)

plt.scatter(x_test[:,0], x_test[:,1], cmap='Paired',c=y_test)


# In[ ]:




