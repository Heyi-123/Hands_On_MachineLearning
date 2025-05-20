####################################################################################################################
#Exercises_Num9: Trainan an SVM classifier on the MINIST dataset. Since SVM classifiers are binary classifiers.
#you will need to use one-versus-the-rest to classify all 10 digits. You may want to tune the hyperparameters using
#small validation sets to speed up the process.
#what accuracy can you reach?
####################################################################################################################

#############################################################################################
#analysis:
#step 1: Load MINIST data  & plot(data)
#step 2: split data
#step 3: Create pipeline and train modes with lineSVC and SVC(kenal=RBF)
#step 4: training modes with lineSVC and SVC(kenal=RBF)
#step 5: evaluate the classifiers perfromance
#step 6: error analysis---find ways to improve it.
##############################################################################################

##############################################################################################
#step 1: Load MINIST data  & plot(data)

from sklearn.datasets import fetch_openml
mnist=fetch_openml("mnist_784",version=1)    #dictionary data structure
mnist.keys()
#print(mnist.keys())
#dict_keys(['data', 'target', 'frame', 'categories', 'feature_names', 'target_names', 'DESCR', 'details', 'url'])

import numpy as np
X,y=mnist['data'],mnist['target']
X=X.to_numpy()
y=y.to_numpy()

#print(X.shape)
#(70000, 784)
#print(y.shape)
#(70000,)

import matplotlib as mpl
import matplotlib.pyplot as plt

some_digit = X[0]
some_digit_image = some_digit.reshape(28,28)

#plt.imshow(some_digit_image,cmap="binary")
#plt.axis('off')
#plt.show()


#print(X[0])


#########################################################################################
#step 2: split data 
#input: X[70000][728], y[70000]
#output: strat_Xtrain_set[60000*0.8][728]
#        strat_Xval_set[60000*0.1][728]
#
            
from sklearn.model_selection import StratifiedShuffleSplit
X_trainVal,X_test,y_trainVal,y_test=X[:60000],X[60000:],y[:60000],y[60000:]

#print(len(X_trainVal))  #len=60000
#print(len(X_test))      #len=10000

split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index, val_index in split.split(X_trainVal,y_trainVal):
    X_train=X_trainVal[train_index]
    y_train=y_trainVal[train_index]
    X_val  =X_trainVal[val_index]
    y_val  =y_trainVal[val_index]

# print(len(X_train)) #48000
# print(len(y_train)) #48000
# print(len(X_val))   #12000
# print(len(y_val))   #12000

# some_digit_X_train = X_train[32]
# some_digit_X_train_image = some_digit_X_train.reshape(28,28)

# print(y_train[32])
# plt.imshow(some_digit_X_train_image,cmap="binary")
# plt.axis('off')
# plt.show()

#############################################################################################
#step 3: Create pipeline and train modes with lineSVC and SVC(kenal=RBF)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm  import LinearSVC

svm_clf=Pipeline([('scaler',StandardScaler()),
                  ('linear_svc',LinearSVC(C=1,loss='hinge')),])
svm_clf.fit(X_train,y_train)




#############################################################################################
#step 4: Predict and evaluate
from sklearn.metrics import accuracy_score,classification_report

y_val_pred=svm_clf.predict(X_val)
print('Accuracy:',accuracy_score(y_val,y_val_pred))

# more specific metrics
print('more specific metrics:',classification_report(y_val,y_val_pred))

# from sklearn.metrics import precision_score,recall_score
# print(precision_score(y_val,y_val_pred))
# print(recall_score(y_val,y_val_pred))

