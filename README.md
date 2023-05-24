# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
~~~
1. Use the standard libraries in python for finding linear regression.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Predict the values of array.
5.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.
6.Obtain the graph.
7.End the program 

~~~
## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: kathirvel.A
RegisterNumber:  212221230047

/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Sirisha Reddy
RegisterNumber: 212222230103

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data=np.loadtxt("/content/ex2data1.txt",delimiter=',')
X=data[:, [0, 1]]
y=data[:, 2]

X[:5]

y[:5]

plt.figure()
plt.scatter(X[y == 1][:, 0], X[y ==1][:, 1], label="Admitted")
plt.scatter(X[y == 0][:, 0], X[y ==0][:, 1], label=" Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

def sigmoid(z):
  return 1 / (1 + np.exp(-z))

plt.plot()
X_plot = np.linspace(-10,10,100)
plt.plot(X_plot, sigmoid(X_plot))
plt.show()

def costFunction(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  j=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
  grad=np.dot(x.T,h-y)/x.shape[0]
  return j,grad
  
X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
j,grad=costFunction(theta,X_train,y)
print(j)
print(grad)

x_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
j,grad=costFunction(theta,X_train,y)
print(j)
print(grad)

def cost(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  j=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
  return j
def gradient(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  grad=np.dot(X.T,h-y)/X.shape[0]
  return grad
X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(X_train,y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
  x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
  y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
  xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
  X_plot=np.c_[xx.ravel(),yy.ravel()]
  X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
  y_plot=np.dot(X_plot,theta).reshape(xx.shape)
  plt.figure()
  plt.scatter(X[y == 1][:, 0], X[y ==1][:, 1], label="Admitted")
  plt.scatter(X[y == 0][:, 0], X[y ==0][:, 1], label=" Not Admitted")
  plt.contour(xx,yy,y_plot,levels=[0])
  plt.xlabel("Exam 1 score")
  plt.ylabel("Exam 2 score")
  plt.legend()
  plt.show()
  
prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta, X):
  X_train=np.hstack((np.ones((X.shape[0],1)),X))
  prob=sigmoid(np.dot(X_train,theta))
  return (prob >= 0.5).astype(int)

np.mean(predict(res.x,X)==y)
*/
```

## Output:

Array Value of x:
![image](https://github.com/KathirvelAIDS/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/94911373/c019cbc2-8eb9-43ea-bbb6-07c2180fdebe)


Array Value of Y:
![image](https://github.com/KathirvelAIDS/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/94911373/ea9aaa49-ce97-40bc-a03c-68b91f69b1d7)


Exam 1 - score graph:

![image](https://github.com/KathirvelAIDS/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/94911373/dc4e2f61-5ed0-4012-993d-c02147e8cdf4)



Sigmoid function graph:

![image](https://github.com/KathirvelAIDS/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/94911373/508d5d7e-9cd4-491b-b39e-1468542f5f4f)



x_train_grad value:
![image](https://github.com/KathirvelAIDS/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/94911373/a3963342-135c-4db0-94e0-47601e5b611d)

y_train_grad value:



![image](https://github.com/KathirvelAIDS/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/94911373/e65aadcb-4af2-4751-8f08-2044549167f2)




Print res.x:

![image](https://github.com/KathirvelAIDS/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/94911373/b6197a0b-f50b-40d8-b75d-f17f2cbefea4)


Decision boundary - graph for exam score:


![image](https://github.com/KathirvelAIDS/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/94911373/f0bbf53b-9df1-471c-9c33-412f47f560a1)




Proability value:




![image](https://github.com/KathirvelAIDS/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/94911373/38ac1667-8037-4db1-bd2f-85acc4165cf7)




Prediction value of mean:



![image](https://github.com/KathirvelAIDS/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/94911373/e6c4ecae-ece8-4532-83ca-83e87327b398)

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

