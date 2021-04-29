#!/usr/bin/env python
# coding: utf-8

# ## Classification with nearest neighbors (k-NN classification).
# <p>In the following, we will work with three fundamental data analysis techniques. We will perform
# classification with the nearest neighbor classifier, a non-linear, non-parametric method for classifica-
# tion. Then we will apply cross-validation for model selection and standard data normalization for
# preprocessing.
# </p>
# 
# ## The data:
# <p>
# The data for the following tasks are taken from a research project financed by Miljfistyrelsen and
# involving researchers from DIKU and PLEN/KU. Selected results from the project are described by
# Rasmussen et al. [2016] and Olsen et al. [2017]. While the problem setting is inspired by Olsen et al.
# [2017], the data were processed differently.
# </p>  
# 
# ## Introduction to the problem:
# 
# <p>   
# Pesticide regulations and a relatively new EU directive on integrated pest management create strong incentives to limit
# herbicide applications. In Denmark, several pesticide action plans have been launched since the late
# 1980s with the aim to reduce herbicide use. One way to reduce the herbicide use is to apply site-specific
# weed management, which is an option when weeds are located in patches, rather than spread uniformly
# over the field. Site-specific weed management can effectively reduce herbicide use, since herbicides are
# only applied to parts of the field. This requires reliable remote sensing and sprayers with individually
# controllable boom sections or a series of controllable nozzles that enable spatially variable applications
# of herbicides. 
# </p>
# <p>
# Preliminary analysis [Rasmussen et al., 2016] indicates that the amount of herbicide use
# for pre-harvest thistle (Cirsium arvense) control with glyphosate can be reduced by at least 60% and
# that a reduction of 80% is within reach. See Figure 1 for an example classification. The problem is
# to generate reliable and cost-effective maps of the weed patches. One approach is to use user-friendly
# drones equipped with RGB cameras as the basis for image analysis and mapping.
# The use of drones as acquisition platform has the advantage of being cheap, hence allowing the
# farmers to invest in the technology. Also, images of suficiently high resolution may be obtained from
# an altitude allowing a complete coverage of a normal sized Danish field in one flight.
# </p>
# 
# 
# <p>
# My data is taken from a number of images of wheat fields taken by a drone carrying a 3K by 4K camera. The 
# ying height was 30 meters. A number of image patches, all showing a
# field area of 3x3 meters were extracted. Approximately half of the patches showed crop, the remaining
# thistles. For each patch only the central 1x1 meter sub-patch is used for performance measurement.
# The full patch was presented to an expert from agriculture and classified as showing either weed (class
# 0) or only crop (class 1).
# For each of the cental sub-patches (here of size 100x100 pixels), 13 rotation and translation invariant
# features were extracted. In more detail, the RGB-values were transformed to HSV and the hue values
# were extracted. The 13 features were obtained by taking a 13-bin histogram of the relevant color
# interval.
# </p>
# 
# ## Reading in the data. 
# <p>
# The training and test data are in the files IDSWeedCropTrain.csv and IDSWeedCropTest.csv, respectively. Each line contains the features and the label for one patch. The last column corresponds to the class label.
# </p>

# ## Exercise 1 (Neartest neighbor classification). 
# 
# <p>
# Apply a nearest neighbor classifier (1-NN) to the data.
# You are encouraged to implement it on your own. However, you can also use scikit-learn.
#     
# <li> 1. Determine the classification accuracy of your model on the training and test data.</li>
#     
# </p>

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier


# In[2]:


dataTrain=np.loadtxt("IDSWeedCropTrain.csv",delimiter="," )
dataTest=np.loadtxt("IDSWeedCropTest.csv", delimiter=",")
XTrain=dataTrain[:,:-1]
YTrain=dataTrain[:,-1]
XTest=dataTest[:,:-1]
YTest=dataTest[:,-1]


# In[3]:


knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(XTrain, YTrain)
accTest = accuracy_score(YTest, knn.predict(XTest))
print(accTest)


# ## Exercise 2 (Cross-validation). 
# 
# <p>
# You are supposed to find a good value for k from [1, 3, 5, 7, 9, 11]. For every choice of k, estimate the performance of the k-NN classifier using 5-fold cross-validation. Pick the k with the lowest average 0-1 loss (classification error), which we will call kbest in the following. Only use the training data in the cross-validation process to generate the folds.  
# 
# <li> 1. Hyperparameter selection using cross-validation </li>
#  
# </p>

# In[4]:


from sklearn.model_selection import KFold


# In[5]:


cv = KFold(n_splits=5)
ks=[1,3,5,7,9,11]
all_accs = []
for k in ks:
    knn=KNeighborsClassifier(n_neighbors=k)
    tot_acc = 0
    for train, test in cv.split(XTrain):
        XTrainCV, XTestCV, YTrainCV, YTestCV = XTrain[train],XTrain[test],YTrain[train],YTrain[test]
        knn.fit(XTrainCV, YTrainCV)
        tot_acc += accuracy_score(YTestCV, knn.predict(XTestCV))
    acc = tot_acc / 5
    all_accs.append(acc)

print(all_accs)


# In[6]:


best_index=np.argmax(all_accs)

k_best=ks[best_index]

print(k_best)


# ## Exercise 3 (Evaluation of classification performance). 
# <p>
# To estimate the generalization performance, build a kbest-NN classifier using the complete training data set IDSWeedCropTrain.csv and evaluate it on the independent test set IDSWeedCropTest.csv .   
# </p>

# In[7]:


k_best_nn = KNeighborsClassifier(n_neighbors=k_best)
k_best_nn.fit(XTrain, YTrain)
best_accTest = accuracy_score(YTest, k_best_nn.predict(XTest))
print(best_accTest)


# ## Exercise 4 (Data normalization). 
# <p>
# Center and normalize the data and repeat the model selection and classification process from Exercise 2 and Exercise 3. However, keep the general rule from above in mind.
# You can implement the normalization yourself. 
#  <li> (i)  Compute the mean and the variance of every input feature (i.e. of every component of the input vector).</li>  
#  <li> (ii) Find the afine linear mapping that transforms the training input data such that the mean and the variance of every    feature are zero and one, respectively, after the transformation.</li> 
# </p>
# <p>Here are three different ways how one could apply the preprocessing
# from scikit-learn, only one of which is correct:
# </p>
# 
# 
# ### version 1
# <p> 
# from sklearn import preprocessing<br> 
# scaler = preprocessing.StandardScaler().fit(XTrain)<br>   
# XTrainN = scaler.transform(XTrain)<br>
# XTestN = scaler.transform(XTest)
# </p>
# 
# ### version 2
# <p>
# from sklearn import preprocessing<br>
# scaler = preprocessing.StandardScaler().fit(XTrain)<br>
# XTrainN = scaler.transform(XTrain)<br>
# scaler = preprocessing.StandardScaler().fit(XTest)<br>
# XTestN = scaler.transform(XTest)
# </p>
# 
# ### version 3
# <p>
# from sklearn import preprocessing<br>
# XTotal = np.concatenate((XTrain,XTest))<br>
# scaler = preprocessing.StandardScaler().fit(XTotal)<br>
# XTrainN = scaler.transform(XTrain)<br>
# XTestN = scaler.transform(XTest)      
# </p>

# In[8]:


from sklearn import preprocessing


# In[9]:


# version 1

scaler = preprocessing.StandardScaler().fit(XTrain)
XTrainN = scaler.transform(XTrain)
XTestN = scaler.transform(XTest)


# In[10]:


print(XTestN)


# In[11]:


print(XTrainN)


# In[12]:


# version 2

scaler = preprocessing.StandardScaler().fit(XTrain)
XTrainN = scaler.transform(XTrain)
scaler = preprocessing.StandardScaler().fit(XTest)
XTestN = scaler.transform(XTest)


# In[13]:


print(XTrainN)


# In[14]:


print(XTestN)


# In[15]:


# version 3

XTotal = np.concatenate((XTrain,XTest))
scaler = preprocessing.StandardScaler().fit(XTotal)
XTrainN = scaler.transform(XTrain)
XTestN = scaler.transform(XTest)


# In[16]:


print(XTrainN)


# In[17]:


print(XTestN)


# In[ ]:




