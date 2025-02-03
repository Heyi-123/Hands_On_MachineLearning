import os
#import tarfile
#import urllib.request 
import pandas as pd


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

#########################################################################################
#way 1st:
def split_train_test(data,test_ratio):
    shuffled_indices=np.random.permutation(len(data)) # output a array with random values belongs to (0,len-1)
    test_set_size=int(len(data)*test_ratio)
    test_indices=shuffled_indices[:test_set_size]
    train_indices=shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]    # operate dataFrame via rowIndex

#train_set, test_set = split_train_test(housing,0.2)    #20250128 hy: remove for the following debug!

#conclusion: This is not perfect! if u run the program again, it will generate a different test set! 
#            Over time, u or your machine learning algorithms will get to see the whole dataset, which is what you want to avoid.
#            what's more, the corresponding solutions will break the next time you fetch an updated dataset.
#Objective: To have a stable train/test split even after updating the dataset, 
#           a common solution is to use each inistance's identifier to decide wether or not it should go in the test set.
#           (assumig instances have a unique and immutable identifier)

####################################################################################################################################
#way 2nd:
housing_with_id=housing.reset_index()     # add a column with original index  to original dataFrame
#print(housing_with_id) #20250128 hy for debug.

from zlib import crc32

def test_set_check(identifier,test_ratio):
    return crc32(np.int64(identifier))&0xffffffff<test_ratio*2**32

def split_train_test_by_id(data,test_ratio,id_column):
    ids=data[id_column]    #dataFrame column operation.
    in_test_set=ids.apply(lambda id_:test_set_check(id_,test_ratio)) #apply the specified function to each element of the column!
    return data.loc[~in_test_set],data.loc[in_test_set]    #dataFrame loc vs iloc!

#train_set,test_set = split_train_test_by_id(housing_with_id,0.2,'index') #20250128 hy: remove for the following debug!
#print(len(train_set))  #20250128 hy for debug
#print(len(housing))    #20250128 hy for debug

#conclusion： if u use th row index as a unique identifier, u need to make sure that new data gets appended to the end of the dataset
#             and that no row ever gets deleted. If this is not possible, then u can try to use the most stable features to build a 
#             unique identifier.

housing_with_id['id']=housing['longitude']+housing['latitude']
#print(housing_with_id)
#train_set,test_set=split_train_test_by_id(housing_with_id, 0.2,'id')  //20250128 hy: remove for the following debug
#print(len(train_set))  #20250128 hy for debug: 20624
#print(len(housing))    #20250128 hy for debug: 20640

#conclusion: combine longitue and latitue to output column 'id' is not feasible!

##################################################################################################################################
#way 3rd:
from sklearn.model_selection import train_test_split 

#train_set,test_set = train_test_split(housing,test_size=0.2,random_state=42) #20250128 hy remove for the following debug.
#print(len(train_set))

##################################################################################################################################
#way 4th
import matplotlib.pyplot as plt
housing['income_cat']=pd.cut(housing['median_income'],bins=[0.,1.5,3.0,4.5,6.,np.inf],labels=[1,2,3,4,5])  #transfer 1 column to user defined column.
#print(housing['income_cat'])
housing['income_cat'].hist()
plt.show()
print(housing['income_cat'].value_counts()/len(housing))

from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

print(strat_test_set['income_cat'].value_counts()/len(strat_test_set))

print(strat_test_set)
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat",axis=1,inplace=True)    # axis=1 to delete column, inplace = true to operate on original dataFrame.
print(strat_test_set)

#summary: (1) purely random sampling methods;
#         (2) stratified random sampling methods.

#input: housing
#output: strat_test_set, strat_train_set