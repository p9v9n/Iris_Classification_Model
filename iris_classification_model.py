#!/usr/bin/env python
# coding: utf-8

# In[5]:


import sys
print("Python version: {}".format(sys.version))
import pandas as pd
print("pandas version: {}".format(pd.__version__))
import matplotlib
print("matplotlib version: {}".format(matplotlib.__version__))
import numpy as np
print("NumPy version: {}".format(np.__version__))
import scipy as sp
print("SciPy version: {}".format(sp.__version__))
import IPython
print("IPython version: {}".format(IPython.__version__))
import sklearn
print("scikit-learn version: {}".format(sklearn.__version__))


# In[6]:


from sklearn.datasets import load_iris
iris_dataset=load_iris()
print("keys of iris dataset:\n",iris_dataset.keys())


# In[7]:


print(iris_dataset['DESCR'][:193]+"\n...")


# In[8]:


print("target names:\n",iris_dataset['target_names'])


# In[9]:


print("feature names:\n",iris_dataset['feature_names'])


# In[6]:


print("type of data:\n",type(iris_dataset['data']))


# In[10]:


print("shape of data:\n",iris_dataset['data'].shape)


# In[8]:


print("first five rows in dataset:\n",iris_dataset['data'][:5])


# In[11]:


print("type of target array:\n",type(iris_dataset['target']))


# In[12]:


print("shape of target:{}".format(iris_dataset['target'].shape))


# In[13]:


print("target:\n",iris_dataset['target'])


# In[14]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(
iris_dataset['data'], iris_dataset['target'], random_state=0)


# In[15]:


print("X_train shape:{}".format(X_train.shape))
print("y_train shape:{}".format(y_train.shape))


# In[16]:


print("X_test shape:{}".format(X_test.shape))
print("y_test shape:{}".format(y_test.shape))


# In[17]:


import pandas as pd
import mglearn
iris_dataframe=pd.DataFrame(X_train, columns=iris_dataset.feature_names)
grr=pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15,15), marker='o',hist_kwds={'bins':20}, s=60, alpha=.8, cmap=mglearn.cm3)


# In[21]:


pip install mglearn


# In[18]:


import pandas as pd
import mglearn
iris_dataframe=pd.DataFrame(X_train, columns=iris_dataset.feature_names)
grr=pd.plotting.scatter_matrix(iris_dataframe, c=y_train, figsize=(15,15), marker='o',hist_kwds={'bins':20}, s=60, alpha=.8, cmap=mglearn.cm3)


# In[20]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=1)


# In[22]:


knn.fit(X_train, y_train)


# In[30]:


X_new=np.array([[5,2.9,1,0.2]])
print("X_new.shape:{}".format(X_new.shape))


# In[31]:


prediction=knn.predict(X_new)
print("Prediction:{}".format(prediction))
print("Predicted terget name:{}".format(iris_dataset['target_names'][prediction]))


# In[32]:


y_pred=knn.predict(X_test)
print("test set prediction:\n{}".format(y_pred))


# In[33]:


print("test set score: {:.2f}".format(np.mean(y_pred==y_test)))


# In[34]:


print("test set score:{:.2f}".format(knn.score(X_test, y_test)))


# In[ ]:




