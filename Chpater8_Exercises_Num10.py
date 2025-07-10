##########################################################################################
#1, use t-SNE to reduce the MNIST dataset down to two dimensions
#2, plot the result using Matplotlib.
#   2.1,you can use a scatterplot using 10 different colors to represent each image's target class.
#   2.2,alternatively, you can replace each dot in the scatterplot with the corresponding instance's class(a digit from 0 to 9)
#   2.3,Or even plot scaled-down versions of digit images themselves.
#   you should get a nice visualization with well-separated clusters of digits.
#3, try using other dimensionality reduction algorithms, and compare the resulting visualizations
#   3.1, PCA;
#   3.2, LLE
#   3.3, MDS

from sklearn.datasets import fetch_openml
mnist=fetch_openml('mnist_784',version=1,as_frame=False)  #FALSE=ARRAY(Numpy), TRUE=Data frame(Panda)
# mnist=fetch_openml('mnist_784',version=1)
mnist.keys()
print(mnist.keys())  #dict_keys(['data', 'target', 'frame', 'categories', 'feature_names', 'target_names', 'DESCR', 'details', 'url'])

import numpy as np
X_org=mnist['data']
X_org=X_org.astype(int)
y_org=mnist['target']
y_org=y_org.astype(int)
print('random sample 3000 instance from datasets :--------')
sample_idx=np.random.choice(len(mnist['data']),size=3000,replace=False)
X_sample=X_org[sample_idx]
y_sample=y_org[sample_idx]
##########################################################################################
#1, use t-SNE to reduce the MNIST dataset down to two dimensions
print('1,use t-SNE to reduce the MNIST dataset down to two dimensions')
##########################################################################################
from sklearn.manifold import TSNE,LocallyLinearEmbedding,MDS
from sklearn.decomposition import PCA

tsne=TSNE(n_components=2,random_state=42)
X_sample_reduced=tsne.fit_transform(X_sample)
print(f'tSNE shape:{X_sample_reduced.shape}')
##################################################################################################################################
#2, plot the result using Matplotlib.
#   2.1,you can use a scatterplot using 10 different colors to represent each image's target class.
#   2.2,alternatively, you can replace each dot in the scatterplot with the corresponding instance's class(a digit from 0 to 9)
#   2.3,Or even plot scaled-down versions of digit images themselves.
print('2,plot the result using Matplotlib.')
###################################################################################################################################
import matplotlib.pyplot as plt
plt.figure(figsize=(10,8))
for digit in range(10):
    mask=(y_sample==digit)
    plt.scatter(X_sample_reduced[mask,0],X_sample_reduced[mask,1],label=str(digit),alpha=0.6)
plt.legend()
plt.title('t-SNE on MNIST 3000 Samples')
plt.show()

   