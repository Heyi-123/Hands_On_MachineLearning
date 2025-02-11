# In this chapter we will be using the MNIST dataset, which is a set of 70,000 small images of digits handwritten by high school
# students and employees of the US Census Bureau.
# This set has been studied so much that it is often called the 'hello world' of Machine Learning.
# ScikIT-Learn provides many helper functions to download popular datasets. 
# MNIST is one of them. 
# The following code fetches the MNIST dataset:
from sklearn.datasets import fetch_openml
mnist=fetch_openml('mnist_784',version=1)     # a dictionary {key1: value1, key2:value2,...}
print(mnist.keys())
#dict_keys(['data', 'target', 'frame', 'categories', 'feature_names', 'target_names', 'DESCR', 'details', 'url'])
# A DESCR key: describing the dataset;
# A data key: containing an array with one row per instance and one column per feature
# A target key: containing an array with labels.

print('--------------------------------------Let us look at these arrays:')
X,y=mnist['data'],mnist['target']
print('dictionary data value(rows, columns) ',X.shape)
print('dictionary target value(rows)',y.shape)


import matplotlib as mpl
import matplotlib.pyplot as plt
some_digit= X[0]
#print(X[0])
'''some_digit_image=some_digit.reshape(28,28)
plt.imshow(some_digit_image,cmap='binary')
plt.axis('off')
plt.show()'''