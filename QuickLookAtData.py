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

#########################################################################################
#quick look 
print(housing.head())   #how does the data looks like?
housing.info()          #check miss data
print(housing['ocean_proximity'].value_counts())  #analyze the specified column's value count
print(housing.describe())                          #analyze all column's feature

import matplotlib.pyplot as plt
housing.hist(bins=50,figsize=(20,15))  #bins=50 means dividing columns into 50 pices during (min,max) as x lable.
plt.show()
