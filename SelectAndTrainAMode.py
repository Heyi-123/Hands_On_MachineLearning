import os
#import tarfile
#import urllib.request 
import pandas as pd
import matplotlib.pyplot as plt


HOUSING_PATH = os.path.join("datasets", "housing")    #datasets\housing


#pd.set_option('display.max_rows', None)     # 设置显示的最大行数为 None，即不限制
pd.set_option('display.max_columns', None)  # 设置显示的最大列数为 None，即不限制
#pd.set_option('display.max_colwidth', 100)  # 设置每列的最大宽度，默认为 50，这里设置为 100

#########################################################################################
##loadData
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path=os.path.join(housing_path,"housing.csv")  #datasets\housing\housing.csv
    return pd.read_csv(csv_path)   

housing=load_housing_data()  
#########################################################################################

import numpy as np

##################################################################################################################################
# stratified random sampling method.
#way 4th
import matplotlib.pyplot as plt
housing['income_cat']=pd.cut(housing['median_income'],bins=[0.,1.5,3.0,4.5,6.,np.inf],labels=[1,2,3,4,5])  #transfer 1 column to user defined column.

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)   #1, stratified;  2, shuffle
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat",axis=1,inplace=True)    # axis=1 to delete column, inplace = true to operate on original dataFrame.

#summary: (1) purely random sampling methods;
#         (2) stratified random sampling methods.

#input: housing
#output: strat_test_set, 
#        strat_train_set

###########################################################################################################
## (train set) =  (numerical) + (!numerical) + (lable)

# Train set --> housing_num,   housing_cat,  lable
# test  set

housing=strat_train_set.drop('median_house_value',axis=1)  # housing + labels= strat_train_set
housing_labels=strat_train_set['median_house_value'].copy() 

housing_num=housing.drop('ocean_proximity',axis=1)         # numerical + !numerical = housing
housing_cat=housing[['ocean_proximity']]   #the result is dataFrame with 1 colomn
#housing_cat=housing['ocean_proximity']    #the result is not dataFrame, just 1 serial array!

############################################################################################################
#data clean---way 1st

#housing_num.info()
#median=housing['total_bedrooms'].median()
#housing_num['total_bedrooms'].fillna(median,inplace=True) #20250131 hy: in first, we have use .info() to identify which column lost values.
#housing_num.info()

#conclusion: Be aware that this is not automatic, pipline will do the jobs.

############################################################################################################
#data clean---way 2nd， not one column by one column, handing all columns.

from sklearn.impute import SimpleImputer

imputer=SimpleImputer(strategy='median')
imputer.fit(housing_num)
#print(imputer.statistics_)          #for debug
#print(housing_num.median().values)  #for debug
#housing_num.info()                  #for debug
X=imputer.transform(housing_num)    #Please be aware, dataFrame --> array (X)
#housing_num.info()                  #for debug
#print(X)                            #for debug

# the result X is a plain NumPy array containning the transformed features.
# if you want to put it back to a pandas DataFrame, it is simple:
housing_tr= pd.DataFrame(X,columns=housing_num.columns,index=housing_num.index)
#print(housing_tr)


##############################################################################################
#Handing test and categorical attributes

#print(housing_cat.head())  #for debug

#so this attribute is a catergory attribute, most ML alogrithms prefer to work with numbers,
#so, let's convert these catergories from test to numbers.
from  sklearn.preprocessing import OrdinalEncoder   #one issue with this representation that 2 nearby values are more similar than 2 distant values
from  sklearn.preprocessing import OneHotEncoder
cat_encoder= OneHotEncoder()
#print(housing_cat) # for debug, housing_cat is a dataFrame with just 1 column.
housing_cat_1hot=cat_encoder.fit_transform(housing_cat)  #dataFrame with just 1 column  --> a SciPy sparse matrix
#print(housing_cat_1hot)
#print(housing_cat_1hot.toarray()) # a SciPy sparse matrix --> full version
#print(cat_encoder.categories_)

#####################################################################################################################################
#Custom transformers
######################################################
#(3) custom transformers
######################################################
from sklearn.base import BaseEstimator, TransformerMixin
rooms_ix,bedrooms_ix,population_ix,households_ix=3,4,5,6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self     #do nothing
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]   #join a,b,c in cloums

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)  #dataFrame-->array
#print(housing)
#print(housing_extra_attribs)

##################################################################################################################################
#Feature scaling
#background: with few exceptions, ML algorithms don't perform well when the input numerical attributes have very different scales
#Note:       note that scaling the target values is generally not required.


##################################################################################################################################
#transformation pipelines

#from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline

num_pipeline=Pipeline([
    ('imputer',SimpleImputer(strategy='median')),  #data clean
    ('attribs_adder',CombinedAttributesAdder()),   #custome transformers
    ('std_scaler',StandardScaler()),               #feature scale---standardization
                       
])
#housing_num_tr=num_pipeline.fit_transform(housing_num)

from sklearn.compose import ColumnTransformer
#prepare column list for numerial handling
num_attribs=list(housing_num)   #equal to   housing_num.columns.tolist()
cat_attribs=['ocean_proximity']
full_pipeline=ColumnTransformer([
    ('num',num_pipeline,num_attribs),    #(data clean) && (custom transformer) && (feature scale)
    ('cat',OneHotEncoder(),cat_attribs), # converter category to numerical
])
housing_prepared=full_pipeline.fit_transform(housing)   
#print(housing_prepared)   # Numpy array.

###################################################################################################################################
#Select and train a mode
###################################################################################################################################

########################################################################
#training and evaluating on the training set

from sklearn.linear_model import LinearRegression

lin_reg=LinearRegression()    
lin_reg.fit(housing_prepared,housing_labels)   #model training

some_data=housing.iloc[:5]  #also dataFrame
some_data_labels=housing_labels.iloc[:5]

somde_data_prepared=full_pipeline.transform(some_data)    #use fit()'s results(dataClean,customTransform,featureScale) to convert(some_data)

print('predictions:',lin_reg.predict(somde_data_prepared))
print('labels:', list(some_data_labels))

# it works, although the predictions are not exactly accurate.
# Let's measure this regression model's RMSE on the whole training set using Scikit-Learn's mean_squared_eror()
################################################################################################################################
# evaluate model's performance
from sklearn.metrics import mean_squared_error
housing_predictions=lin_reg.predict(housing_prepared)
lin_mse=mean_squared_error(housing_labels,housing_predictions)
lin_rmse=np.sqrt(lin_mse)
print('linearRegression error:',lin_rmse)

# This is better than nothing, but clearly not a great score.
# most districts' median_housing_values range between $120,000 and $265,000, so a typical prediction error of $68,628 is not very satisfaction.
# This is an example of a model underfitting the trainning data.
# When this happens it can mean that the features do not provide enough information to make good predictions,
# or that the model is not powerful enough.

# the main ways to fix underfitting are:
# (1) to select a more powerful model
# (2) to feed the training algorithm with better features
# (3) to reduce restraints on the model.

#This model is not regularized, which rules out the last option.
#you could try to add more features(e.g., the log of population),
#but first let's try a more complex model to see how it does.

#let's train a DecisionTreeRegresor.  
#Thisis a powerful model, capable of finding complex nonlinear relationships in the data.
from sklearn.tree import DecisionTreeRegressor
tree_reg=DecisionTreeRegressor()
tree_reg.fit(housing_prepared,housing_labels)

#Now that the model is trained, let's evaluate it on the trainning set:
housing_predictions=tree_reg.predict(housing_prepared)
tree_mse=mean_squared_error(housing_labels,housing_predictions)
tree_rmse=np.sqrt(tree_mse)
print('DecisionTree error:',tree_rmse)

#wait, what! No error at all? Could this model really be absolutely perfect?
#Of course, it is muh more likely that the model has badly overfit the data.
# How can you be sure?
# As we saw earlier, you don't want to touch the test set until you are ready to launch a model you are confident about, 
# so, you need to use part of training set for training and part of it for model validation.

###############################################################################################################################
#Better evaluation using cross-validation
#Note that: the training_set already be prepared before cross validation.

print('-------------------------Decision Tree: CrossValidationScore-----------------------------------------------------')
from sklearn.model_selection import cross_val_score
scores=cross_val_score(tree_reg,housing_prepared,housing_labels,scoring='neg_mean_squared_error',cv=10)
#analysis:
#(1) split training_set into 10 folds;
#(2) train mode with 9 folds, 
#(3) pick the rest one fold for evaluation,
tree_rmse_scores=np.sqrt(-scores)
def display_scores(scores):
    print('scores:',scores)
    print('Mean:',scores.mean())
    print('standard deviation:',scores.std())

display_scores(tree_rmse_scores)
# Now the Decision Tree doesn't look as good as it did earlier.
# In fact, it seems to perform worse than the Linear Regression model!
# Notice that cross-validation allows you to get not only an estimate of performance of your model, 
# but also a measure of how precise this estimate is (i.e., its standard deviation)

# but, cross validation comes at the cost of training the model several times, so it is not always possible.
print('-------------------------Linear Regression: CrossValidationScore-----------------------------------------------------')
#Let's compute the same scores for Linear Regression model just to be sure.
lin_scores=cross_val_score(lin_reg,housing_prepared,housing_labels,scoring='neg_mean_squared_error',cv=10)
lin_rmse_scores=np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)

# That's right, the Decision Tree is overfitting so badly that it performs worse than the Linear Regression model.

# Let't try one last mode now: random Forest
print('-------------------------Random Forest: CrossValidationScore-----------------------------------------------------')

from sklearn.ensemble import RandomForestRegressor
forest_reg= RandomForestRegressor()
forest_reg.fit(housing_prepared,housing_labels)
forest_scores=cross_val_score(forest_reg,housing_prepared,housing_labels,scoring='neg_mean_squared_error',cv=10)
forest_rmse_scores=np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)

# Wow, this is much better: Random forrests look very promising.
# However, note that the score on the training set is still much lower than on the validation sets, meaning that
#          the mode is still overfitting the training set.
# Possible solutions for overfitting are:
#          (1) Simplify the model;
#          (2) Constrain it(i.e., regularize it)
#          (3) Get a lot more training data
# Before you dive much deeper into Random forests, however, you should try out many other models from various categories of ML (e.g., SVM, ...)
#          without spending too much time tweaking the hyperparameters.
#          The goal is to shortlist a few (2 or 5) promising models.






