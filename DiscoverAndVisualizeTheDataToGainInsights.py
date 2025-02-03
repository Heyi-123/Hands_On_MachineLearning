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

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
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
housing=strat_train_set.copy()
#print(len(housing))

housing.plot(kind="scatter",x='longitude',y='latitude')
#plt.show()
#conclusion: this looks like california all right, but other than that it is hard to see any particular pattern.
#            Setting the alpha option to 0.1 makes it esier to visualize the places where there is a high density of data points
housing.plot(kind="scatter",x='longitude',y='latitude',alpha=0.1)    # 20250128 hy: ensity can be identified.
#our brains are very good at spotting patterns in pictures, but you may need to play around with visualization parameters to make the 
#patterns stand out.
#s : the radius of each circle ,  radius --> district population
#c : color,  color --> price
#jet:  from blue to red 
housing.plot(kind='scatter',x='longitude',y='latitude',alpha=0.4,s=housing['population']/100,label='population',figsize=(10,7),
             c='median_house_value',cmap=plt.get_cmap('jet'),colorbar=True,  )
plt.legend
plt.show()

###############################################################################################################
#looking for correlations
housing_num=housing.drop('ocean_proximity',axis=1)         
corr_matrix=housing_num.corr()
#print(corr_matrix)
print(corr_matrix['median_house_value'].sort_values(ascending=False))






















