####################################################################################################################
#Exercises_Num7: Train and fine tune a Decision Tree for the moons dataset by following these steps:
#a, Use make_moons(n_samples=10000,noise=0.4) to generate a moons dataset.
#b, Use train_test_split() to split the dataset into a training set and a test set.
#c, Use grid search with cross-validation (with the help of the GridSearchCV class) to find good hyperparameter values 
#     for a DecisionTreeClassifier.
#     Hint:try various values for max_leaf_nodes.
#, Train it on the full training set using these hyperparameters, and measure your model's performance on the test set.
#     You should get roughly 85% to 87% accuracy.
####################################################################################################################



from sklearn.datasets import make_moons
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

#generate make moon dataset
X,y=make_moons(n_samples=10000,noise=0.4)

plt.scatter(X[:,0],X[:,1],c=y)
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()


#Split datasets
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#(traing)&&(Find the best estimator)
param_grid={'max_leaf_nodes':[5,10,15,20,25]}
dtc=DecisionTreeClassifier(random_state=42)
grid_search=GridSearchCV(dtc,param_grid,cv=5)
grid_search.fit(X_train,y_train)


#evaluate the classifiers
best_dtc=grid_search.best_estimator_
# print(best_dtc)
best_dtc.fit(X_train,y_train)
y_pred=best_dtc.predict(X_test)
accuracy=accuracy_score(y_test,y_pred)
print(accuracy)

# # confusion matrix
# from sklearn.metrics import confusion_matrix
# import seaborn as sns
# cm=confusion_matrix(y_val,y_val_pred)
# # visualization confusion matrix
# plt.figure(figsize=(10,8))
# sns.heatmap(cm,annot=True,fmt='d',cmap='Blues')
# plt.xlabel('Predicted Label')
# plt.ylabel('True Label')
# plt.title('Confusion Matrix')
# plt.show()



# mnist=fetch_openml("mnist_784",version=1)    #dictionary data structure
# mnist.keys()
# #print(mnist.keys())
# #dict_keys(['data', 'target', 'frame', 'categories', 'feature_names', 'target_names', 'DESCR', 'details', 'url'])

# import numpy as np
# X,y=mnist['data'],mnist['target']
# X=X.to_numpy()
# y=y.to_numpy()

# #print(X.shape)
# #(70000, 784)
# #print(y.shape)
# #(70000,)

# import matplotlib as mpl
# import matplotlib.pyplot as plt

# some_digit = X[0]
# some_digit_image = some_digit.reshape(28,28)

# #plt.imshow(some_digit_image,cmap="binary")
# #plt.axis('off')
# #plt.show()


# #print(X[0])


# #########################################################################################
# #step 2: split data 
# #input: X[70000][728], y[70000]
# #output: strat_Xtrain_set[60000*0.8][728]
# #        strat_Xval_set[60000*0.1][728]
# #
            
# from sklearn.model_selection import StratifiedShuffleSplit
# X_trainVal,X_test,y_trainVal,y_test=X[:60000],X[60000:],y[:60000],y[60000:]

# #print(len(X_trainVal))  #len=60000
# #print(len(X_test))      #len=10000

# split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
# for train_index, val_index in split.split(X_trainVal,y_trainVal):
#     X_train=X_trainVal[train_index]
#     y_train=y_trainVal[train_index]
#     X_val  =X_trainVal[val_index]
#     y_val  =y_trainVal[val_index]

# # print(len(X_train)) #48000
# # print(len(y_train)) #48000
# # print(len(X_val))   #12000
# # print(len(y_val))   #12000

# # some_digit_X_train = X_train[32]
# # some_digit_X_train_image = some_digit_X_train.reshape(28,28)

# # print(y_train[32])
# # plt.imshow(some_digit_X_train_image,cmap="binary")
# # plt.axis('off')
# # plt.show()

# #############################################################################################
# #step 3: Create pipeline and train modes with lineSVC and SVC(kenal=RBF)
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
# from sklearn.svm  import LinearSVC

# svm_clf=Pipeline([('scaler',StandardScaler()),
#                   ('linear_svc',LinearSVC(C=1,loss='hinge')),])
# svm_clf.fit(X_train,y_train)




# #############################################################################################
# #step 4: Predict and evaluate
# from sklearn.metrics import accuracy_score,classification_report

# y_val_pred=svm_clf.predict(X_val)
# print('Accuracy:',accuracy_score(y_val,y_val_pred))

# # more specific metrics
# print('more specific metrics:',classification_report(y_val,y_val_pred))

# # from sklearn.metrics import precision_score,recall_score
# # print(precision_score(y_val,y_val_pred))
# # print(recall_score(y_val,y_val_pred))


# # confusion matrix
# from sklearn.metrics import confusion_matrix
# import seaborn as sns
# cm=confusion_matrix(y_val,y_val_pred)
# # visualization confusion matrix
# plt.figure(figsize=(10,8))
# sns.heatmap(cm,annot=True,fmt='d',cmap='Blues')
# plt.xlabel('Predicted Label')
# plt.ylabel('True Label')
# plt.title('Confusion Matrix')
# plt.show()

