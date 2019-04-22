import numpy as np

"""
    Minigratch Gradient Descent Function to train model
    1. Format the data
    2. call four_nn function to obtain losses
    3. Return all the weights/biases and a list of losses at each epoch
    Args:
        epoch (int) - number of iterations to run through neural net
        w1, w2, w3, w4, b1, b2, b3, b4 (numpy arrays) - starting weights
        x_train (np array) - (n,d) numpy array where d=number of features
        y_train (np array) - (n,) all the labels corresponding to x_train
        num_classes (int) - number of classes (range of y_train)
        shuffle (bool) - shuffle data at each epoch if True. Turn this off for testing.
    Returns:
        w1, w2, w3, w4, b1, b2, b3, b4 (numpy arrays) - resulting weights
        losses (list of ints) - each index should correspond to epoch number
            Note that len(losses) == epoch
    Hints:
        Should work for any number of features and classes
        Good idea to print the epoch number at each iteration for sanity checks!
        (Stdout print will not affect autograder as long as runtime is within limits)
"""
def minibatch_gd(epoch, w1, w2, w3, w4, b1, b2, b3, b4, x_train, y_train, num_classes, shuffle=True):

    #IMPLEMENT HERE

    return w1, w2, w3, w4, b1, b2, b3, b4, losses

"""
    Use the trained weights & biases to see how well the nn performs
        on the test data
    Args:
        All the weights/biases from minibatch_gd()
        x_test (np array) - (n', d) numpy array
        y_test (np array) - (n',) all the labels corresponding to x_test
        num_classes (int) - number of classes (range of y_test)
    Returns:
        avg_class_rate (float) - average classification rate
        class_rate_per_class (list of floats) - Classification Rate per class
            (index corresponding to class number)
    Hints:
        Good place to show your confusion matrix as well.
        The confusion matrix won't be autograded but necessary in report.
"""
def test_nn(w1, w2, w3, w4, b1, b2, b3, b4, x_test, y_test, num_classes):

    avg_class_rate = 0.0
    class_rate_per_class = [0.0] * num_classes
    return avg_class_rate, class_rate_per_class

"""
    4 Layer Neural Network
    Helper function for minibatch_gd
    Up to you on how to implement this, won't be unit tested
    Should call helper functions below
"""
def four_nn(W1, W2, W3, W4, b1, b2, b3, b4, X, Y, num_classes): """X = x_train, Y = y_train"""
    Z1, acache1 = affine_forward(X,W1,b1)
    A1, rcache1 = relu_forward(Z1)
    Z2, acache2 = affine_forward(A1,W2,b2)
    A2, rcache2 = relu_forward(Z2)
    Z3, acache3 = affine_forward(A2,W3,b3)
    A3, rcache3 = relu_forward(Z3)
    F, acache4 = affine_forward(A3,W4,b4)
    if (test == True):"""modify this condition"""
        classifications = np.zeros(F.shape[0])
        for i in range(F.shape[0]):
            classifications[i] = max(F[i])
        return classifications
    loss, dF = cross_entropy(F,Y)
    dA3, dW4, db4 = affine_backward(dF,acache4)
    dZ3 = relu_backward(dA3,rcache3)
    dA2, dW3, db3 = affine_backward(dZ3,acache3)
    dZ2 = relu_backward(dA2,rcache2)
    dA1, dW2, db2 = affine_backward(dZ2,acache2)
    dZ1 = relu_backward(dA1,rcache1)
    dX, dW1, db1 = affine_backward(dZ1,acache1)
    eta = 0.1
    W1 = W1 - eta*dW1
    W2 = W2 - eta*dW2
    W3 = W3 - eta*dW3
    W4 = W4 - eta*dW4
    return loss,W1, W2, W3, W4  """may need modify this line"""
    pass

"""
    Next five functions will be used in four_nn() as helper functions.
    All these functions will be autograded, and a unit test script is provided as unit_test.py.
    The cache object format is up to you, we will only autograde the computed matrices.

    Args and Return values are specified in the MP docs
    Hint: Utilize numpy as much as possible for max efficiency.
        This is a great time to review on your linear algebra as well.
"""
def affine_forward(A, W, b):
    
    """Z=np.dot(A,W)+ b
    """
    Z=np.zeros((A.shape[0],W.shape[1]))
    for i in range(A.shape[0]):
        for j in range(W.shape[1]):
            Z[i][j]+=b[j]
            for k in range(A.shape[1]):
                Z[i][j]+=A[i][k]*W[k][j]
    cache = (A,W,b)
    return Z, cache

def affine_backward(dZ, cache):
    A,W,b = cache[0], cache[1], cache[2]
    """dA = np.dot(dZ,W.T)
    dW = np.dot(A.T,dZ)
    db = np.dot(np.ones(i),Z)"""
    for i in range(A.shape[0]):
        for k in range(A.shape[1]):
            for j in range(Z.shape[1]):
                dA[i][k] += dZ[i][j]*W[k][j]
                dW[k][j] += A[i][k]*dZ[i][j]
                db[j] += dZ[i][j]
    return dA, dW, dB

def relu_forward(Z):
    cache = Z
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            A[i][j] = max(elem,0)
    return A, cache

def relu_backward(dA, cache):
    Z = cache
    dZ = dA
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            dZ[i][j] = dZ
    return dZ

def cross_entropy(F, y):
    loss = np.zeros(F.shape[0])
    dF = np.zeros(F.shape)
    l1 = np.zeros(F.shape[0])
    l2 = np.zeros(F.shape[0])
    for i in range(F.shape[0]):
        l1[i] += F[i][y[i]]
        for k in range(F.shape[1]):
            l2[i] += exp(F[i][k])
        for j in range(F.shape[1]):
            isLabel = j==y[i]
            dF[i][j] = -1/n(isLabel-np.exp(F[i][j])/l2[i])
        loss[i] = -1/n(l1[i]-np.log(l2[i]))
    
            
    return loss, dF

