####################################################################################################################
#Exercises_Num8: 
#     1,Load the MINIST data(introduced in Chapter 3), 
#     2,split it into a trainig set, a validation set, and a test set
#       (e.g., use 50,000 instances for training,)
#              use 10,000 for validation, 
#              and 10,000 for testing.
#     3,Then train various classifiers, such as:
#              a random forest classifier
#              an extra-Trees classifier,
#              and an SVM classifier.
#     4,Next, try to combine them into an ensemble that 
#       outperforms each individual classifier on the validation set, using soft or hard voting.
#     5,Once you have found one, try it on the test set.
#       How much better does it perform compared to the individual classifier.
####################################################################################################################

#################################################################################################################################
#     1,Load the MINIST data(introduced in Chapter 3), 
#################################################################################################################################
# from sklearn.datasets import make_moon
from sklearn.datasets import fetch_openml
mnist=fetch_openml('mnist_784',version=1,as_frame=False)  #FALSE=ARRAY(Numpy), TRUE=Data frame(Panda)
# mnist=fetch_openml('mnist_784',version=1)
mnist.keys()
print(mnist.keys())  #dict_keys(['data', 'target', 'frame', 'categories', 'feature_names', 'target_names', 'DESCR', 'details', 'url'])

################################################################################################################################
#     2,split it into a trainig set, a validation set, and a test set
#       (e.g., use 50,000 instances for training,)
#              use 10,000 for validation, 
#              and 10,000 for testing.
#################################################################################################################################
from sklearn.model_selection import StratifiedShuffleSplit

OrgData=mnist['data']  #(70000, 784)
OrgLabel=mnist['target']
# print(OrgData.shape)  

split=StratifiedShuffleSplit(n_splits=1,test_size=10000,random_state=42)
for train_full_idx, test_idx in split.split(OrgData,OrgLabel):
    X_test,y_test=OrgData[test_idx],OrgLabel[test_idx]   #10,000 for test set
    X_trainFull,y_trainFull=OrgData[train_full_idx],OrgLabel[train_full_idx]
    split_val=StratifiedShuffleSplit(n_splits=1,test_size=10000,random_state=42)
    for train_idx, val_idx in split.split(X_trainFull,y_trainFull):
        X_train,y_train=X_trainFull[train_idx],y_trainFull[train_idx]   #50,000 for train set
        X_val,    y_val=X_trainFull[val_idx]  ,y_trainFull[val_idx]     #10,000 for validation set


print(f'1,train set shape is {X_train.shape}')
print(f'2,validation set shape is {X_val.shape}')
print(f'3,test set shape is {X_test.shape}')

################################################################################################################################
#     3,Then train various classifiers, such as:
#              a random forest classifier
#              an extra-Trees classifier,
#              and an SVM classifier.
################################################################################################################################
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.svm import LinearSVC,SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.model_selection import cross_val_score,KFold
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix




#################################################################################
#(3.1) LinearSVC
#################################################################################
print('3.1,linearSVC start training, prediction, and evaluation:--------')
# lineSVC_clf=Pipeline([('scaler',StandardScaler()),
#                             ('svm_clf',LinearSVC(C=1,loss='hinge'))])
lineSVC_clf=Pipeline([('scaler',StandardScaler()),
                            ('svm_clf',SVC(kernel='linear',C=1,probability=True))])

# cv=KFold(n_splits=3,shuffle=True,random_state=42)
# rbf_kernal_svm_clf_accuracy=cross_val_score(estimator=rbf_kernal_svm_clf,X=X_train,y=y_train,cv=cv,scoring='accuracy')

#Not used cross validation with CV=5, time complexity is huge.!!
lineSVC_clf.fit(X_train,y_train)
y_val_svc_pre=lineSVC_clf.predict(X_val)

print(f'3.1,svc accuracy score is: {accuracy_score(y_val,y_val_svc_pre)}')
print(f'3.1,svc clssification report is: {classification_report(y_val,y_val_svc_pre)}')




#################################################################################
#(3.2) RandomForest
#################################################################################
print('3.2,random frorest start training, prediction, and evaluation:--------')
rf_clf=Pipeline([('scaler',StandardScaler()),
                            ('rf_clf',RandomForestClassifier(n_estimators=500,min_samples_split=10,n_jobs=-1))])
rf_clf.fit(X_train,y_train)
y_val_rf_pre=rf_clf.predict(X_val)

print(f'3.2, random frorest accuracy score is: {accuracy_score(y_val,y_val_rf_pre)}')
print(f'3.2, random frorest clssification report is: {classification_report(y_val,y_val_rf_pre)}')

#################################################################################
#(3.3) Extra-RandomForest
#################################################################################
print('3.3, extra random frorest start training, prediction, and evaluation:--------')
et_clf=Pipeline([('scaler',StandardScaler()),
                            ('et_clf',ExtraTreesClassifier(n_estimators=500,min_samples_split=10,n_jobs=-1))])
et_clf.fit(X_train,y_train)
y_val_extra_rf_pre=et_clf.predict(X_val)

print(f'3.3, extra random frorest accuracy score is: {accuracy_score(y_val,y_val_extra_rf_pre)}')
print(f'3.3, extra random frorest clssification report is: {classification_report(y_val,y_val_extra_rf_pre)}')


####################################################################################################################
#     4,Next, try to combine them into an ensemble that 
#       outperforms each individual classifier on the validation set, using soft or hard voting.
#     5,Once you have found one, try it on the test set.
#       How much better does it perform compared to the individual classifier.
####################################################################################################################
from sklearn.ensemble import VotingClassifier

hard_volting=VotingClassifier(estimators=[('svc',lineSVC_clf),
                                          ('rf',rf_clf),
                                          ('et',et_clf)],voting='soft')  #hard --> soft
hard_volting.fit(X_train,y_train)
y_val_hv_pre=hard_volting.predict(X_val)

print(f'4.0, hard volting ensemble accuracy score is: {accuracy_score(y_val,y_val_hv_pre)}')
print(f'4.0, hard volting ensemble clssification report is: {classification_report(y_val,y_val_hv_pre)}')




