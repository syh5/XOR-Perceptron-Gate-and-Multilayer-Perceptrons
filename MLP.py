import os 
import numpy as np
import pandas as pd  
import matplotlib.pyplot as plt  
import sys

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


def computeCost(X,y,theta,reg):
    N = len(y)
    _,probs,_,_ = predict(X,theta)
    cost = -np.mean(np.log(probs[range(N),y]))+0.5*reg*(np.sum(np.square(theta[0]))+np.sum(np.square(theta[2])))
    return cost		
			
def computeGrad(X,y,theta,reg): # returns nabla
    W = theta[0]
    b = theta[1]
    W2 = theta[2]
    b2 = theta[3]
    N = len(X)
    hpre = np.dot(X,W) + b
    
    _,_,df,df2 = predict(X,theta)
    
    df[range(N),y] -=1
    df /= N
    
    df2[range(N),y] -=1
    df2 /= N
    df2 = np.dot(df2,np.transpose(W2))
    
    for i in range(hpre.shape[0]):
        for j in range(hpre.shape[1]):
            if(hpre[i][j] < 0):
                df2[i][j] = 0

    dW = np.dot(np.transpose(X),df2) + reg*W
    db = np.sum(df2, axis=0)    
    
    hpre = np.dot(X,W) + b
    h = np.maximum(0,hpre)
    dW2 = np.dot(np.transpose(h),df)+ reg*W2
    db2 = np.sum(df, axis=0)
    return (dW,db,dW2,db2)

def predict(X,theta):
    W = theta[0]
    b= theta[1]
    W2 = theta[2]
    b2 = theta[3]
    
    hpre = np.dot(X,W) + b
    h = np.maximum(0,hpre)
    f = np.dot(h,W2) + b2
	# compute the class probabilities
    probs = np.exp(f)/np.sum(np.exp(f),axis=1,keepdims=True)
    
    df = np.array(probs,copy=True)

    df2 = np.array(probs,copy=True)
    
    return (f,probs,df,df2)
	
np.random.seed(0)
# Load in the data from disk
path = os.getcwd() + '/data/spiral_train.dat'  
data = pd.read_csv(path, header=None) 

# set X (training data) and y (target variable)
cols = data.shape[1]  
X = data.iloc[:,0:cols-1]  
y = data.iloc[:,cols-1:cols] 

# convert from data frames to numpy matrices
X = np.array(X.values)  
y = np.array(y.values)
y = y.flatten()

# initialize parameters randomly
D = X.shape[1]
K = np.amax(y) + 1

# initialize parameters randomly
h = 100 # size of hidden layer
W = 0.01 * np.random.randn(D,h)
b = np.zeros((1,h))
W2 = 0.01 * np.random.randn(h,K)
b2 = np.zeros((1,K))
theta = (W,b,W2,b2)

# some hyperparameters
n_e = 1000
check = 100 # every so many pass/epochs, print loss/error to terminal
step_size = 1e-0
reg = 1e-3 # regularization strength

# gradient descent loop
for i in range(n_e):
    b = theta[1]
    W = theta[0]
    
    W2 = theta[2]
    b2 = theta[3]
    dW,db,dW2,db2 = computeGrad(X,y,theta,reg)
    loss = computeCost(X,y,theta,reg)
    if i % check == 0:
        print ("iteration %d: loss %f" % (i, loss))

	# perform a parameter update
    b = b - step_size*db
    W = W - step_size*dW
    b2 = b2 - step_size*db2
    W2 = W2 - step_size*dW2
    
    theta = (W,b,W2,b2)


scores, probs,_,_ = predict(X,theta)
predicted_class = np.argmax(scores, axis=1)
print ('training accuracy: %.2f' % (np.mean(predicted_class == y)))

# plot the resulting classifier
h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
					 
Z, P,_,_ = predict(np.c_[xx.ravel(), yy.ravel()], theta)

Z = np.argmax(Z, axis=1)
Z = Z.reshape(xx.shape)
fig = plt.figure()
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
#fig.savefig('spiral_net.png')

plt.show()