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
