

import numpy as np
import matplotlib.pyplot as plt


print('-----------------------------normal equation---------------------------------------')
#Create instances & target
X=2*np.random.rand(100,1)
y=4+3*X +np.random.randn(100,1)
#print(y)

#plot to see the relationship between instances X and target y
plt.scatter(X,y)
plt.title('Figure 4-1 Randomly generated linear dataset')
plt.xlabel('x1')
plt.ylabel('y')
plt.show()

X_b = np.c_()


