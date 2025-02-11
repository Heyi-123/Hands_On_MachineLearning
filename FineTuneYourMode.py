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
        
    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            raise ValueError('input_features must be provided')
        new_feature_names=list(input_features)
        if self.add_bedrooms_per_room:
            new_feature_names.extend(['rooms_per_household','population_per_household','bedrooms_per_room'])
        else:
            new_feature_names.extend(['rooms_per_household','population_per_household'])
        return np.array(new_feature_names)
         
    

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


from sklearn.ensemble import RandomForestRegressor


print('-----------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------Fine-Tune your mode-----------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------')

##########################################################################################################################
#Fine-Tune your mode----Grid search
# One option would be to fiddle with hyperparameters manually, until you find great combination of hyperparameter values.
# This would be very tedious work, and you may not have time to expore many combinations.
# Instead, you should get Scikit-Learn't GridSearchCV to search for you.
# All you need to do is tell it:
#  (1) which hyperparameters you want it to experiment with;
#  (2) what values to try out;
#  (3) it will use cross-validation to evaluate all the possible combinations of hyperparameter values.

from sklearn.model_selection import GridSearchCV
param_grid= [
        { 'n_estimators':[3,10,30],
          'max_features':[2,4,6,8]
        },
        { 'bootstrap':[False],
          'n_estimators':[3,10],
          'max_features':[2,3,4]
        },
] 
forest_reg= RandomForestRegressor()
grid_search=GridSearchCV(forest_reg,param_grid,cv=5,scoring='neg_mean_squared_error',return_train_score=True)
grid_search.fit(housing_prepared,housing_labels)
print(grid_search.best_params_)

print(grid_search.best_estimator_)
#and of course the evaluaion scores are also available:
cvres=grid_search.cv_results_
for mean_score, params in zip(cvres['mean_test_score'],cvres['params']):
    print(np.sqrt(-mean_score),params)
#in this example, we obtain the best solution by setting the max_features hyperparameter to 8 and the n_estimators hyperparameter to 30.
#the RMSE score for this combination is 49,682, which is slightly better than the score you got earlier using the default hyperparameter values(which was 50,182)


print('-----------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------Randomized search-------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------')
# when the hyperparameter search space is large, it is often preferable to use RandomizedSearchCV instead.
# Instead of trying out all possible combinations, it evaluates a given number of random combinations by selecting a random
# value for each hyperparameter at every iteration.

print('-----------------------------------------------------------------------------------------------------------------')
print('-----------------------------------------Ensemble Methods--------------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------')
# Another way to fine-tune your system is to try to combine the models that perform best.
# The group will often perfrom better than the best individual model(just like Random Forests perform better than the individual 
# Decision Trees thy rely on), especially if the individual models make very different types of errors.

print('-----------------------------------------------------------------------------------------------------------------')
print('------------------------Analyze the Best Models and their Errors-------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------')
# (1) see if possible to drop useless features;
# (2) analyze errors

# you will often gain good insights on the problem by inspecting the best models.
# For example, the ramdomForestRegressor can indicate the relative imprtance of each attribute for making accurate predictions:
feature_importances = grid_search.best_estimator_.feature_importances_
print(feature_importances)
#[7.72745516e-02 7.22718969e-02 4.07646508e-02 1.64165154e-02
# 1.70152824e-02 1.86706828e-02 1.69912714e-02 3.20909363e-01
# 6.51189348e-02 1.05494299e-01 7.81599363e-02 9.86039061e-03
# 1.49707211e-01 7.71179869e-06 5.20673974e-03 6.13056266e-03]

#Let's display these importance socres next to their corresponding attribute names:
#print(full_pipeline.get_feature_names_out())

print(full_pipeline.get_feature_names_out())

#['num__longitude' 
# 'num__latitude' 
# 'num__housing_median_age'
# 'num__total_rooms' 
# 'num__total_bedrooms'
# 'num__population'
# 'num__households'
# 'num__median_income' 
# 'num__rooms_per_household'
# 'num__population_per_household' 
# 'num__bedrooms_per_room'
# 'cat__ocean_proximity_<1H OCEAN' 
# 'cat__ocean_proximity_INLAND'
# 'cat__ocean_proximity_ISLAND' 
# 'cat__ocean_proximity_NEAR BAY'
# 'cat__ocean_proximity_NEAR OCEAN']

extra_attribs=['rooms_per_hhold','pop_per_hhold','bedrooms_per_room']
cat_encoder_transformer=full_pipeline.named_transformers_['cat']  # get cat's transformer
cat_one_hot_attribs=list(cat_encoder_transformer.categories_[0])
attributes=num_attribs + extra_attribs + cat_one_hot_attribs
print(sorted(zip(feature_importances,attributes),reverse=True))


# (1) with this information , you may want to dry dropping some of the less useful features.
# (2) you should also look at the specific errors that your system makes, then try to understand why it makes them 
#     and what could fix the problem (adding extra features or getting rid of uninformatives ones, cleanning up outliers,etc) 

print('-----------------------------------------------------------------------------------------------------------------')
print('------------------------Evaluate our system on the test set -----------------------------------------------------')
print('-----------------------------------------------------------------------------------------------------------------')
# After tweaking your models for a while, you eventually have a system that performs sufficiently well.
# Now it the time to evaluate the final model on the test set.
# (1) Get the predictors and the labels from your test set.
# (2) Run your full_pipeline to transform the data;
# (3) Evaluate the final model on the test set.
from sklearn.metrics import mean_squared_error

final_model = grid_search.best_estimator_   #get the predictors

X_test = strat_test_set.drop('median_house_value',axis=1) # =1, delete column,   dataFrame
y_test = strat_test_set['median_house_value'].copy() # numpy series
x_test_prepared = full_pipeline.transform(X_test)

final_predictions = final_model.predict(x_test_prepared)
final_mse= mean_squared_error(y_test,final_predictions)
final_rmse=np.sqrt(final_mse)
print(final_rmse)


#you can compute a 95% confidence interval for the generalization error using scipy.status.t.interval()
from scipy import stats
confidence = 0.95
squared_errors = (final_predictions - y_test)**2
print(np.sqrt(stats.t.interval(confidence,len(squared_errors)-1,
                         loc=squared_errors.mean(),
                         scale=stats.sem(squared_errors)))
)


################################################
#launch, monitor, and maintain your system
#(1)  deploy your model on the web server,  via web API
#(2)  deploy your model on the cloud,       via web API
#(3)  Deploy is not the end of the story, you alos need to write monitoring code to check your system's live performance at regular intervals and 
#     trigger alerts when it drops.
#     (3.1) If the data keeps evolving, ou will need to update our datasets and 
#            retrain your model regularly. you should probably automate the whold process as much as possible.
#     (3.2) you should also make sure you evaluate the model's input data quality.
#     (3.3) finally, make sure you keep backups of every model you create and have process and tool in place to roll back to a previous model quickly,
#           in case the new model starts falling badly for some reason.