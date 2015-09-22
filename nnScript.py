import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import cPickle as pickle

def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W
    
    
    
def sigmoid(z):
    
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""
    y=1/(1+np.exp(-1*z))
    return y
    #your code here
    
    

def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - divide the original data set to training, validation and testing set
           with corresponding labels
     - convert original data set from integer to double by using double()
           function
     - normalize the data to [0, 1]
     - feature selection"""
    
    mat = loadmat('D:/Study/Courses/574/PA1/basecode/mnist_all.mat') #loads the MAT object as a Dictionary
    
    #Pick a reasonable size for validation data
    
    
    #Your code here
    train_data=np.empty(shape=[0, 784])
    train_label = np.array([])
    validation_data = np.empty(shape=[0, 784])
    validation_label = np.array([])
    test_data = np.empty(shape=[0, 784])
    test_label = np.array([])
    
    for i in range(0, 10):
        testTemp=mat.get('test'+str(i))
        testTemp=np.double(testTemp)
        testTemp=testTemp/255
        testLabel=np.empty(testTemp.shape[0])
        testLabel.fill(i)
        temp=mat.get('train'+str(i))
        Arr=range(temp.shape[0])
        aperm=np.random.permutation(Arr)
        A1=temp[aperm[0:1000],:]
        A1=np.double(A1)
        A1=A1/255
        A2=temp[aperm[1000:],:]
        A2=np.double(A2)
        A2=A2/255
        vLabel=np.empty(A1.shape[0])
        vLabel.fill(i)
        tLabel=np.empty(A2.shape[0])
        tLabel.fill(i)
        train_data=np.append(train_data,A2,axis=0)
        validation_data=np.concatenate([validation_data,A1])
        validation_label=np.concatenate([validation_label,vLabel])
        train_label=np.concatenate([train_label,tLabel])
        test_data=np.append(test_data,testTemp,axis=0)
        #test_data=np.concatenate([test_data,testTemp])
        test_label=np.concatenate([test_label,testLabel])
    
    return train_data, train_label, validation_data, validation_label, test_data, test_label
    
    
    

def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""
    
    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args
    w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0  
    
    #Your code here
    #
    #
    #
    #
    #
    newTrain_label=np.empty(shape=[training_data.shape[0],n_class])
    newTrain_label.fill(0.0)
    for i in range(0,training_data.shape[0]):
        for k in range(0,n_class):
            if(training_label[i]==k):
                newTrain_label[i][k]=1.0
    biasColumn=np.empty([training_data.shape[0],1])
    biasColumn.fill(1)
    training_data=np.append(training_data,biasColumn,axis=1)
    tempA=np.empty(shape=[training_data.shape[0],n_hidden])
    tempA.fill(0)
    tempB=np.empty(shape=[training_data.shape[0],n_class])
    tempB.fill(0)
    tempZ=np.empty(shape=[training_data.shape[0],n_hidden+1])
    tempZ.fill(0)
    tempO=np.empty(shape=[training_data.shape[0],n_class])
    tempO.fill(0)
    w1Transpose=w1.transpose()
    w2Transpose=w2.transpose()
    tempA=np.dot(training_data,w1Transpose)
    tempZ=sigmoid(tempA)
    biasColumn=np.empty([tempZ.shape[0],1])
    biasColumn.fill(1)
    tempZ=np.append(tempZ,biasColumn,axis=1)
    tempB=np.dot(tempZ,w2Transpose)
    tempO=sigmoid(tempB)
    oneMinusNewTrain_label=1-newTrain_label
    oneMinusTempO=1-tempO
    logTempO=np.log(tempO)
    logOneMinusTempO=np.log(oneMinusTempO)
    
    obj_val=0.0
    for i in range(0,training_data.shape[0]):
        #obj_val+=newTrain_label[i][l]*logTempO[i][l]+(1-newTrain_label[i][l])*logOneMinusTempO[i][l]
        obj_val+=np.dot(newTrain_label[i],np.transpose(logTempO[i]))+np.dot((1-newTrain_label[i]),np.transpose(logOneMinusTempO[i]))
    obj_val=obj_val*-1
    obj_val=obj_val/training_data.shape[0]
    
    
    grad_w1=np.empty(w1.shape)#check whether array is having values 0
    grad_w2=np.empty(w2.shape)
    grad_w1.fill(0.0)
    grad_w2.fill(0.0)
    tempDelta=tempO-newTrain_label
    tempDeltaTranspose=np.transpose(tempDelta)
    grad_w2=np.dot(tempDeltaTranspose,tempZ)
    grad_w2=grad_w2+lambdaval*w2
    grad_w2=grad_w2/training_data.shape[0]
    
    oneMinusZJ=1-tempZ
    multZJ=oneMinusZJ*tempZ
    deltaMulW2=np.dot(tempDelta,w2)
    multZJ=multZJ*deltaMulW2
    multZJ=np.delete(multZJ,n_hidden,1)
    multZJ=np.transpose(multZJ)
    grad_w1=np.dot(multZJ,training_data)
    grad_w1=grad_w1+lambdaval*w1
    grad_w1=grad_w1/training_data.shape[0]
    
    
    
    #Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    #you would use code similar to the one below to create a flat array
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)

    #obj_grad = np.array([])
    
    return (obj_val,obj_grad)



def nnPredict(w1,w2,data):
    
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels""" 
    
    labels = np.empty([])
    #Your code here
    
    #biasColumn=np.empty([data.shape[0],1])
    #biasColumn.fill(1)
    #data=np.append(data,biasColumn,axis=1)
    #tempA=np.empty(shape=[data.shape[0],n_hidden])
    #tempA.fill(0)
    #tempB=np.empty(shape=[data.shape[0],n_class])
    #tempB.fill(0)
    #tempZ=np.empty(shape=[data.shape[0],n_hidden+1])
    #tempZ.fill(0)
    #tempO=np.empty(shape=[data.shape[0],n_class])
    #tempO.fill(0)
    #w1Transpose=w1.transpose()
    #w2Transpose=w2.transpose()
    #tempA=np.dot(data,w1Transpose)
    #tempZ=sigmoid(tempA)
    #biasColumn=np.empty([tempZ.shape[0],1])
    #biasColumn.fill(1)
    #tempZ=np.append(tempZ,biasColumn,axis=1)
    #tempB=np.dot(tempZ,w2Transpose)
    #tempO=sigmoid(tempB)
    #for i in range(0,data.shape[0]):
    #    maximum=tempO[i][0]
    #    for l in range(0,n_class):
    #        if maximum<tempO[i][l]:
    #            maximum=tempO[i][l]
    #    labels[i]=maximum
    #return labels
    
    biasColumn=np.empty([data.shape[0],1])
    biasColumn.fill(1)
    data=np.append(data,biasColumn,axis=1)
    tempA=np.empty(shape=[data.shape[0],n_hidden])
    tempA.fill(0)
    tempB=np.empty(shape=[data.shape[0],n_class])
    tempB.fill(0)
    tempZ=np.empty(shape=[data.shape[0],n_hidden+1])
    tempZ.fill(0)
    tempO=np.empty(shape=[data.shape[0],n_class])
    tempO.fill(0)
    w1Transpose=w1.transpose()
    w2Transpose=w2.transpose()
    tempA=np.dot(data,w1Transpose)
    tempZ=sigmoid(tempA)
    biasColumn=np.empty([tempZ.shape[0],1])
    biasColumn.fill(1)
    tempZ=np.append(tempZ,biasColumn,axis=1)
    tempB=np.dot(tempZ,w2Transpose)
    tempO=sigmoid(tempB)
    labels=np.argmax(tempO,axis=1)
    
    return labels



"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data,validation_label, test_data, test_label = preprocess();

#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]; 

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 50;

# set the number of nodes in output unit
n_class = 10;				   

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)

# set the regularization hyper-parameter
lambdaval = 1.0;

args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter' : 50}    # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)



#In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
#and nnObjGradient. Check documentation for this function before you proceed.
#nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

t=(n_hidden,w1,w2,lambdaval)
with open('params.pickle','wb')as params:
    p=pickle.Pickler(params)
    p.dump(t)
#pickle.dump(nn_params, open("params.pickle","wb"))
#Test the computed parameters

predicted_label = nnPredict(w1,w2,train_data)

#find the accuracy on Training Dataset
print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1,w2,validation_data)

#find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')


predicted_label = nnPredict(w1,w2,test_data)

#find the accuracy on Validation Dataset

#predicted_label= np.double(predicted_label)
print('\n Test set Accuracy:' + str(100*np.mean((predicted_label == test_label).astype(float))) + '%')
