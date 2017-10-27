
import numpy as np
import logging
import json
from utility import * #custom methods for data cleaning

FILE_NAME_TRAIN = 'train.csv' #replace this file name with the train file
FILE_NAME_TEST = 'test.csv' #replace
ALPHA = 1e-1
EPOCHS = 6000#keep this greater than or equl to 5000 strictly otherwise you will get an error
MODEL_FILE = 'models/model1'
t1 = True
t = False
logging.basicConfig(filename='output.log',level=logging.DEBUG)

np.set_printoptions(suppress=True)
#################################################################################################
##################################### write the functions here ####################################
#################################################################################################
#this function appends 1 to the start of the input X and returns the new array
def appendIntercept(X):
    #steps
    a = np.ones((X.shape[0],1))
    X = np.hstack((a, X))
    return X 

 #intitial guess of parameters (intialize all to zero)
 #this func takes the number of parameters that is to be fitted and returns a vector of zeros
def initialGuess(n_thetas):
    return np.zeros(n_thetas)
    

def train(theta, X, y, model):
     J = [] 
     m = len(y)
     for i in range(EPOCHS):
	p = (predict(X,theta))
        c = costFunc(m,y,p)
	J.append(c)
        cg = calcGradients(X,y,p,m)
	theta = makeGradientUpdate(theta, cg)
     
     #your  gradient descent code goes here
     #steps
     #run you gd loop for EPOCHS that you have defined
        #calculate the predicted y using your current value of theta
        # calculate cost with that current theta using the costFunc function
        #append the above cost in J
        #calculate your gradients values using calcGradients function
        # update the theta using makeGradientUpdate function (don't make a new variable assign it back to theta that you received)

     model['J'] = J
     model['theta'] = list(theta)
     return model

#this function will calculate the total cost and will return it
def costFunc(m,y,y_predicted):
    return -(np.sum(y*np.log(y_predicted)+(1-y)*np.log(1-y_predicted))*1.0/m)

def calcGradients(X,y,y_predicted,m):
    diff = y_predicted-y
    a = (np.sum(diff.reshape(diff.shape[0],1)*X , axis=0)*1.0)/m
    return a

#this function will update the theta and return it
def makeGradientUpdate(theta, grads):
    return theta - (ALPHA *grads)

#this function will take two paramets as the input
def predict(X,theta):
    a= (1/(1+np.exp(-X.dot(theta))))
    return a


######################## main function ###########################################
def main():
    if(t1):
        model = {}
        X_df,y_df = loadData(FILE_NAME_TRAIN)
        X,y, model = normalizeData(X_df, y_df, model)
        X = appendIntercept(X)
        theta = initialGuess(X.shape[1])
        model = train(theta, X, y, model)
        with open(MODEL_FILE,'w') as f:
            f.write(json.dumps(model))
	accuracy(X,y,model,FILE_NAME_TRAIN)
    else:
        model = {}
        with open(MODEL_FILE,'r') as f:
            model = json.loads(f.read())
            X_df, y_df = loadData(FILE_NAME_TEST)
            X,y = normalizeTestData(X_df, y_df, model)
            X = appendIntercept(X)
            accuracy(X,y,model,FILE_NAME_TEST)

if __name__ == '__main__':
    main()

