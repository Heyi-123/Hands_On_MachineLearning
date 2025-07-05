####################################################################################################################
#Exercises_Num9: 
#   1, run the individual classifiers from the previous exercise to make predictions on the validation set,
#   2, and create a new training set with the resulting predictions:
#           each training instance is a vector containing the set of predictions from all your classifiers for an image,
#           and the target is the image's class.
#   3, train a classifier on this new training set.
#   congradulations, you have just trained a blender, and together with the classifiers it forms a stacking ensemble!
#   4, Now, evaluate the ensemble on the test set:
#           (4.1) make predictions with all your classifiers,
#           (4.2) then feed the predictions to the blender to get the blender to get the ensemble's predictions.
#           (4.3) how does it compare to the volting classifier you trained earlier?
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
