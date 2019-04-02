#!/usr/bin/env python
# coding: utf-8

# # Assignment 10

# ## Ian Kurzrock

# ### 1. Issue relevant _import_ commands (for ___numpy, pandas, sklearn___).

# In[1]:


import numpy as np
import pandas as pd
import sklearn as skl

from sklearn import cluster


# ### 2. Using the appropiate pandas function, read the _diabetes.txt_ into a dataframe.

# In[2]:


data = pd.read_csv('diabetes.txt', delimiter=' ', header=None)


# In[3]:


isinstance(data, pd.DataFrame)


# ### 3. Using the appropriate pandas function, make the features dataframe.

# In[4]:


data_x = data.iloc[:,0:8]


# In[5]:


data_y = data.iloc[:][8]


# ### 4. numpy requires numerical arrays, make the numerical features from the dataframe of 3

# In[6]:


data_x = data_x.as_matrix()


# In[7]:


data_y = data_y.as_matrix()


# ### 5. Make 2 Spectral Cluster objects

# In[8]:


spectral = cluster.SpectralClustering(n_clusters=2, eigen_solver='arpack')


# ### 6. Use _fit_predict()_ function to do a _spectral clustering_ of the data from the matrix 4.

# In[9]:


C = spectral.fit_predict(data_x, data_y)


# ### 7. Comparing the clusters produced by 6. and ground-truth classes in the dataset

# #### a. Extract the dataframe of the classes

# In[10]:


classes = data.iloc[:,8]


# #### b. Using the replace() function of pandas, replace 'class' with '' to that class1 or 0 becomes just 1 or 0. Use the regex expression argument.

# In[11]:


classes = classes.replace(to_replace='class', value='', regex=True)


# #### c. Using to_numeric(), transform entries '0' and '1' into number

# In[12]:


classes = pd.to_numeric(classes)


# #### d. transform into a matrix

# In[13]:


classes = classes.as_matrix()


# #### e. Using equal() and sum() of numpy, count how many entries of d matrix and matrix C actually match.

# In[14]:


np.sum(np.equal(classes, C))/758.0

