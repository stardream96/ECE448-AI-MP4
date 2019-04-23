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
    batch_size = 200
    losses = []         #a list of total loss at each epoch. Length = epoch
    for e in range(epoch):  #or (1, epoch + 1). not sure
        print("epoch " + str(e + 1))

        # One very common error is that you are probably shuffling your x_train
        # but not your y_train corresponding to the correct rows. Or you are not loading in batches properly.
        # MODIFY SHUFFLE
        if shuffle == True:
            # use advanced indexing to create unison-shuffled arrays
            p = np.random.permutation(len(x_train))
            x_train = x_train[p]
            y_train = y_train[p]

        total_loss = 0

        #add "+1" in range function to pass unit test
        for i in range(1, int(len(x_train)/batch_size) + 1): #number of examples / batch size
            x = x_train[(i - 1) * batch_size  :  i * batch_size]    # split x & y into smaller batches
            y = y_train[(i - 1) * batch_size  :  i * batch_size]
            #print("four_nn called for the " + str(i) + " times")
            loss, w1, w2, w3, w4 = four_nn(w1, w2, w3, w4, b1, b2, b3, b4, x, y, num_classes, test = False)
            total_loss += loss
        losses.append(total_loss)
    print(losses)
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
    classifications = four_nn(w1, w2, w3, w4, b1, b2, b3, b4, x_test, y_test, num_classes, test = True)
    print("test_nn")
    print(len(classifications))
    print(len(y_test))
    print(classifications)
    print(y_test)
    count = [0.0] * num_classes
    class_rate_per_class = [0.0] * num_classes
    avg_class_rate = 0.0
    for i in range(len(y_test)):
        correct_label = y_test[i]
        count[correct_label] += 1
        if correct_label == classifications[i]:
            class_rate_per_class[correct_label] += 1
            avg_class_rate += 1.0
    total = 0
    for i in range(num_classes):
        total += count[i]
        class_rate_per_class[i] /= count[i]
    avg_class_rate /= total
    return avg_class_rate, class_rate_per_class


"""
    4 Layer Neural Network
    Helper function for minibatch_gd
    Up to you on how to implement this, won't be unit tested
    Should call helper functions below
"""


def four_nn(W1, W2, W3, W4, b1, b2, b3, b4, X, Y, num_classes, test):
    """X = x_train, Y = y_train"""
    Z1, acache1 = affine_forward(X,W1,b1)
    A1, rcache1 = relu_forward(Z1)
    Z2, acache2 = affine_forward(A1,W2,b2)
    A2, rcache2 = relu_forward(Z2)
    Z3, acache3 = affine_forward(A2,W3,b3)
    A3, rcache3 = relu_forward(Z3)
    F, acache4 = affine_forward(A3,W4,b4)

    #print("four_nn halfway done")

    if (test == True):
        classfications = [np.argmax(x) for x in F]
        # classifications = np.zeros(F.shape[0])
        # for i in range(F.shape[0]):
        #     classifications[i] = np.argmax(F[i])
        return classfications

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

    return loss, W1, W2, W3, W4

"""
    Next five functions will be used in four_nn() as helper functions.
    All these functions will be autograded, and a unit test script is provided as unit_test.py.
    The cache object format is up to you, we will only autograde the computed matrices.
    Args and Return values are specified in the MP docs
    Hint: Utilize numpy as much as possible for max efficiency.
        This is a great time to review on your linear algebra as well.
"""


def affine_forward(A, W, b):

    """
    Z = np.zeros((A.shape[0], W.shape[1]))
    for i in range(A.shape[0]):
        for j in range(W.shape[1]):
            Z[i][j] += b[j]
            for k in range(A.shape[1]):
                print("check if the program is stuck here")
                Z[i][j] += A[i][k] * W[k][j]
    """
    Z = np.dot(A, W) + b        #dot product of 2 arrays
    cache = (A, W, b)
    return Z, cache


def affine_backward(dZ, cache):
    A, W, b = cache[0], cache[1], cache[2]
    dA = np.dot(dZ,W.T)
    dW = np.dot(A.T,dZ)
    dB = sum(dZ)
    """
    for i in range(A.shape[0]):
        for k in range(A.shape[1]):
            for j in range(Z.shape[1]):
                dA[i][k] += dZ[i][j] * W[k][j]
                dW[k][j] += A[i][k] * dZ[i][j]
                db[j] += dZ[i][j]
    """
    return dA, dW, dB


def relu_forward(Z):
    # was using for loops but too slow
    A = np.maximum(Z, 0)    # Compare two arrays and returns a new array containing the element-wise maxima
    cache = Z
    return A, cache


def relu_backward(dA, cache):
    # dZij = 0 if Zij = 0. else dZij = dAij.

    Z = cache
    dZ = dA
    """
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            dZ[i][j] = dZ
    """
    dZ[Z <= 0] = 0
    return dZ


def cross_entropy(F, y):
    loss = 0
    dF = np.zeros(F.shape)
    l1 = np.zeros(F.shape[0])
    l2 = np.zeros(F.shape[0])
    n = F.shape[0]      # size of F and n
    for i in range(F.shape[0]):
        temp = int(y[i])
        l1[i] += F[i][temp]     # use temp to avoid weird compile error
        for k in range(F.shape[1]):
            l2[i] += np.exp(F[i][k])
        for j in range(F.shape[1]):
            isLabel = j == y[i]
            dF[i][j] = (-1 / n) * (isLabel - np.exp(F[i][j]) / l2[i])
        loss += (-1 / n) * (l1[i] - np.log(l2[i]))

    return loss, dF
