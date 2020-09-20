# -*- coding: utf-8 -*-
"""
@author: jhhalls
"""

# import the libraries
#image processing
import os
import numpy as np
import pandas as pd
from sklearn import linear_model 
from sklearn import model_selection
from sklearn import metrics
import matplotlib.pyplot as plt
import statsmodels.api as sm
import matplotlib

# load the data
os.chdir("f:/datafiles")  
train=pd.read_csv("mnist_train.csv",header=None)
test=pd.read_csv("mnist_test.csv",header=None)


# have a look at the data
x_train=train.iloc[:,1:]        #0th column is label which contains the no. for which this row data we have, so removed
y_train=train.iloc[:,0]         #0th of ytrain will tell us what no. is this actually
x_test=test.iloc[:,1:] 
y_test=test.iloc[:,0]
y_train5=(ytrain==5).astype(np.int)
y_test5=(ytest==5).astype(np.int)

## 0th row of the data set/ first image
# reshape the matrix to have a look at the image
arow=x_train.iloc[0,:].reshape(28,28)       #this will signify one digit in whole, taking one complete row at a time
# show the image
plt.imshow(arow,cmap=matplotlib.cm.binary,interpolation='nearest')
# cross check with the output
y_train[0]


# train the model
sgd=linear_model.SGDClassifier(random_state=42)
sgd.fit(x_train,y_train5)
y_train5.value_counts()
predicted=sgd.predict(x_train)

sgd1=linear_model.SGDClassifier(random_state=42)
sgd1.fit(x_test,y_test5)
ytest5.value_counts()
predicted1=sgd1.predict(x_test)


## Evaluate the matrix
# using confusion matrix
conf_mat=metrics.confusion_matrix(ytest5,predicted1)
print(conf_mat)

yscore=sgd.decision_function(xtest)
predicted2=(yscore>=0).astype(np.int) #by putting constraint on y-score we can manage threshold limit
print(predicted2)
conf_mat1=metrics.confusion_matrix(ytest5,predicted1)
print(conf_mat1)


#accuracy=96.3%   precission=75.1%  


