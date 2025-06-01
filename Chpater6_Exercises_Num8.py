####################################################################################################################
#Exercises_Num8: Train and fine tune a Decision Tree for the moons dataset by following these steps:
#a, Use make_moons(n_samples=10000,noise=0.4) to generate a moons dataset.
#b, Use train_test_split() to split the dataset into a training set and a test set.
#c, Use grid search with cross-validation (with the help of the GridSearchCV class) to find good hyperparameter values 
#     for a DecisionTreeClassifier.
#     Hint:try various values for max_leaf_nodes.
#, Train it on the full training set using these hyperparameters, and measure your model's performance on the test set.
#     You should get roughly 85% to 87% accuracy.
####################################################################################################################
#8. Grow a forest by following these steps:
#a. Continuing the previous exercise, generate 1,000 subsets of the training set, 
#       each containing 100 instances selected randomly. 
#       Hint: you can use Scikit - Learn’s ShuffleSplit class for this.
# b. Train one Decision Tree on each subset, using the best hyperparameter values found in the previous exercise. 
#       Evaluate these 1,000 Decision Trees on the test set. Since they were trained on smaller sets, 
#       these Decision Trees will likely perform worse than the first Decision Tree, achieving only about 80% accuracy.
# c. Now comes the magic. For each test set instance, generate the predictions of the 1,000 Decision Trees, 
#       and keep only the most frequent prediction (you can use SciPy’s mode() function for this). 
#       This approach gives you majority - vote predictions over the test set.
# d. Evaluate these predictions on the test set: 
#       you should obtain a slightly higher accuracy than your first model (about 0.5 to 1.5% higher).
# Congratulations, you have trained a Random Forest classifier!
#################################################################################################################################


from sklearn.datasets import make_moons
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,GridSearchCV,StratifiedShuffleSplit,ShuffleSplit
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

# ss=ShuffleSplit(n_splits=1000,test_size=100,random_state=42)
# for _, subset_index in ss.split(X,y):
#     X_train=X_trainVal[train_index]
#     y_train=y_trainVal[train_index]
#     X_val  =X_trainVal[val_index]
#     y_val  =y_trainVal[val_index]


split=StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)
for train_index, val_index in split.split(X_trainVal,y_trainVal):
    X_train=X_trainVal[train_index]
    y_train=y_trainVal[train_index]
    X_val  =X_trainVal[val_index]
    y_val  =y_trainVal[val_index]




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
