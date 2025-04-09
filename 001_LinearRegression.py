

import numpy as np
import matplotlib.pyplot as plt

print('-----------------------------------------------------------------------------------------')
print('------------------Linear Regression: solution 1 start, normal equation-------------------')
print('-----------------------------------------------------------------------------------------')
#Create instances & target
X=2*np.random.rand(100,1)
y=4+3*X +np.random.randn(100,1)
#print(X)

#plot to see the relationship between instances X and target y
'''plt.scatter(X,y)
plt.title('Figure 4-1 Randomly generated linear dataset')
plt.xlabel('x1')
plt.ylabel('y')
plt.show()  #linear mode'''

#Objective: Normal equation to solve the optimal model theta
#input:  X_b, y
#Output: model para theta_Best
#Note: X_b * theta_Best = y
X_b = np.c_[np.ones((100,1)),X]  # add x0=1 to each instance, total 100 instances  (x0, x1)
theta_Best=np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
print(f'we can get theta_Best = {theta_Best}')
#we can get theta_Best = [[4.5259157 ] [2.63742433]]

#Objective: Create a instance to predict via the model already calculated.
X_new = np.array([[0],[2]])
X_new_b = np.c_[np.ones((2,1)),X_new]
y_predict = X_new_b.dot(theta_Best)
print(f"we can predict target= {y_predict}")
#we can predict taget= [[3.98798196] [9.95731857]]

#plot this model's predictions
plt.title('Figure 4-2 Linear Regression model predictions')
plt.xlabel('x1')
plt.ylabel('y')
plt.plot(X_new,y_predict,'r--')
plt.plot(X,y,'b.')
plt.axis([0,2,0,15])
plt.show()

print('-----------------------------------------------------------------------------------------')
print('------------------Linear Regression: solution 1 ending, normal equation------------------')
print('-----------------------------------------------------------------------------------------')



print('------------------------------------------------------------------------------------------------------------')
print('------------------Normal equation vs Linear Regression class training & prediction for comparasion----------')
print('------------------------------------------------------------------------------------------------------------')
from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,y)
y_predict_lin_reg=lin_reg.predict(X_new)
print(f"we can predict target= {y_predict_lin_reg}")
#we can predict target= [[ 3.98845218] [10.0065591 ]]

print('------------------------------------------------------------------------------------------------------------')
print('------------------pseudoinverse solution: not introduced here-----------------------------------------------')
print('------------------------------------------------------------------------------------------------------------')



