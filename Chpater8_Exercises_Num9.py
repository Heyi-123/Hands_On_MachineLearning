

####################################################################################################################
#Chapter 8 dimensionality reduction, exercises_num9
#1, Load the MNIST dataset (introduced in Chapter 3) 
#2, split it into a training set and a test set 
#      use the first 60,000 instances for training,
#      use the remaining 10,000 for testing
#3, train a random forest classifer on the dataset and time how long it takes
#4, then evaluate the resultinig mode on the test set.
#5, Next, use PCA to reduce the dadasets' dimensionality, with an explained variance ration of 95%
#6, train a new random forest classifier on the reduced dataset and see how long it takes.
#7, Was training much faster? 
#8, Next, evaluate the classifier on the test set. How does it compare to the previous classifier?
######################################################################################################################




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
#2, split it into a training set and a test set 
#      use the first 60,000 instances for training,
#      use the remaining 10,000 for testing
print('2,split it into a training set and a test set :--------')
#################################################################################################################################
from sklearn.model_selection import StratifiedShuffleSplit

Xorg=mnist['data']  #(70000, 784)
y_org=mnist['target']
# print(OrgData.shape)  

split=StratifiedShuffleSplit(n_splits=1,test_size=10000,random_state=42)
for train_idx, test_idx in split.split(Xorg,y_org):
    X_test,y_test=Xorg[test_idx],y_org[test_idx]        #10,000 for test set
    X_train,y_train=Xorg[train_idx],y_org[train_idx]    #60,000 for train set


print(f'2.1,train set shape is {X_train.shape}')
print(f'2.2,test set shape is {X_test.shape}')

################################################################################################################################
#3, train a random forest classifer on the dataset and time how long it takes
print('3,random frorest start training, prediction, and evaluation:--------')
################################################################################################################################
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
# from sklearn.svm import LinearSVC,SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
# import numpy as np
# from sklearn.model_selection import cross_val_score,KFold
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
import time
from datetime import datetime





rdf_clf=Pipeline([('scaler',StandardScaler()),
                 ('rf_clf',RandomForestClassifier(n_estimators=500,min_samples_split=10,n_jobs=-1))])

start_time=time.time()
start_dt=datetime.now()

rdf_clf.fit(X_train,y_train)

end_time=time.time()
end_dt=datetime.now()

duration_time=end_time-start_time
duration_dt=end_dt-start_dt
print(f'Random forest train time:{duration_time}')  
print(f'Random forest train time:{duration_dt}')
#################################################################################
#4, then evaluate the resultinig mode on the test set.
print('4,then evaluate the resultinig mode on the test set:--------')
#################################################################################
y_test_pre=rdf_clf.predict(X_test)

print(f'Random forest evaluation matrix:-------------------')
print(f'(1), random frorest accuracy score is: {accuracy_score(y_test,y_test_pre)}')
print(f'(2), random frorest clssification report is: {classification_report(y_test,y_test_pre)}')

##################################################################################################################
#5, Next, use PCA to reduce the dadasets' dimensionality, with an explained variance ration of 95%
print('5,Next, use PCA to reduce the dadasets dimensionality, with an explained variance ration of 95%:--------')
##################################################################################################################
from sklearn.decomposition import PCA
pca=Pipeline([('scaler',StandardScaler()),
              ('rf_clf',PCA(n_components=0.95))])
X_reduced=pca.fit_transform(X_train)

##################################################################################################################
#6, train a new random forest classifier on the reduced dataset and see how long it takes.
print('6,train a new random forest classifier on the reduced dataset and see how long it takes:--------')
##################################################################################################################
pca_rdf_clf=RandomForestClassifier(n_estimators=500,min_samples_split=10,n_jobs=-1)


pca_start_dt=datetime.now()
pca_rdf_clf.fit(X_reduced,y_train)
pca_end_dt=datetime.now()

pca_duration_dt=pca_end_dt-pca_start_dt 
print(f'pca+Random forest train time:{pca_duration_dt}')

##################################################################################################################
#7, Was training much faster? 
##################################################################################################################

##################################################################################################################
#8, Next, evaluate the classifier on the test set. How does it compare to the previous classifier?
##################################################################################################################
X_test_reduced=pca.transform(X_test)
y_test_pre_pca=pca_rdf_clf.predict(X_test_reduced)

print(f'Random forest evaluation matrix:-------------------')
print(f'(1), random frorest accuracy score is: {accuracy_score(y_test,y_test_pre_pca)}')
print(f'(2), random frorest clssification report is: {classification_report(y_test,y_test_pre_pca)}')
# #################################################################################
# #(3.1) LinearSVC
# #################################################################################
# print('3.1,linearSVC start training, prediction, and evaluation:--------')
# # lineSVC_clf=Pipeline([('scaler',StandardScaler()),
# #                             ('svm_clf',LinearSVC(C=1,loss='hinge'))])
# lineSVC_clf=Pipeline([('scaler',StandardScaler()),
#                             ('svm_clf',SVC(kernel='linear',C=1,probability=True))])

# # cv=KFold(n_splits=3,shuffle=True,random_state=42)
# # rbf_kernal_svm_clf_accuracy=cross_val_score(estimator=rbf_kernal_svm_clf,X=X_train,y=y_train,cv=cv,scoring='accuracy')

# #Not used cross validation with CV=5, time complexity is huge.!!
# lineSVC_clf.fit(X_train,y_train)
# y_val_svc_pre=lineSVC_clf.predict(X_val)

# print(f'3.1,svc accuracy score is: {accuracy_score(y_val,y_val_svc_pre)}')
# print(f'3.1,svc clssification report is: {classification_report(y_val,y_val_svc_pre)}')




# #################################################################################
# #(3.2) RandomForest
# #################################################################################
# print('3.2,random frorest start training, prediction, and evaluation:--------')
# rf_clf=Pipeline([('scaler',StandardScaler()),
#                             ('rf_clf',RandomForestClassifier(n_estimators=500,min_samples_split=10,n_jobs=-1))])
# rf_clf.fit(X_train,y_train)
# y_val_rf_pre=rf_clf.predict(X_val)

# print(f'3.2, random frorest accuracy score is: {accuracy_score(y_val,y_val_rf_pre)}')
# print(f'3.2, random frorest clssification report is: {classification_report(y_val,y_val_rf_pre)}')

# #################################################################################
# #(3.3) Extra-RandomForest
# #################################################################################
# print('3.3, extra random frorest start training, prediction, and evaluation:--------')
# et_clf=Pipeline([('scaler',StandardScaler()),
#                             ('et_clf',ExtraTreesClassifier(n_estimators=500,min_samples_split=10,n_jobs=-1))])
# et_clf.fit(X_train,y_train)
# y_val_extra_rf_pre=et_clf.predict(X_val)

# print(f'3.3, extra random frorest accuracy score is: {accuracy_score(y_val,y_val_extra_rf_pre)}')
# print(f'3.3, extra random frorest clssification report is: {classification_report(y_val,y_val_extra_rf_pre)}')


# ####################################################################################################################
# #     4,Next, try to combine them into an ensemble that 
# #       outperforms each individual classifier on the validation set, using soft or hard voting.
# #     5,Once you have found one, try it on the test set.
# #       How much better does it perform compared to the individual classifier.
# ####################################################################################################################
# from sklearn.ensemble import VotingClassifier

# hard_volting=VotingClassifier(estimators=[('svc',lineSVC_clf),
#                                           ('rf',rf_clf),
#                                           ('et',et_clf)],voting='soft')  #hard --> soft
# hard_volting.fit(X_train,y_train)
# y_val_hv_pre=hard_volting.predict(X_val)

# print(f'4.0, hard volting ensemble accuracy score is: {accuracy_score(y_val,y_val_hv_pre)}')
# print(f'4.0, hard volting ensemble clssification report is: {classification_report(y_val,y_val_hv_pre)}')




