import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
    dW = np.zeros(W.shape) # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in xrange(num_train):
        scores = X[i,:].dot(W)
        correct_class_score = scores[y[i]]
        diff_count = (scores - correct_class_score + 1)>0
        ###########
        for j in xrange(num_classes):
            margin = scores[j] - correct_class_score + 1 # note delta = 1
            if j == y[i]:
                dW[:,j] += -np.sum(np.delete(diff_count,j))*X[i,:]
                continue
            dW[:,j] += diff_count[j]*X[i]
            if margin > 0:
                   loss += margin
            
    
    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train
    # Add regularization to the loss.
    dW += reg*W
    loss += reg * np.sum(W * W)

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape) # initialize the gradient as zero
    #num_train = X.shape[1]
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
    num_classes = W.shape[1]
    num_train = X.shape[0]   
    ################################
    loss = 0.0
    scores = X.dot(W)               #500 X 10 dimensional array. 
    correct_class_score = scores[np.arange(num_train),y]
    margins = np.maximum(0,scores - correct_class_score[:,np.newaxis] + 1)
    margins[np.arange(num_train),y] = 0
    loss = np.sum(margins)
    loss /= num_train
    loss += reg*np.sum(W*W)
    
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

    
  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
    #X_mask = np.zeros(margins.shape)
    #X_mask[margins>0] = 1
    #incorrect_counts = np.sum(X_mask,axis=1)
    #X_mask[np.arange(num_train),y] = -incorrect_counts
    #incorrect_counts = 0
    #dW = (X.T).dot(X_mask)
    #dW /= num_train
    #dW += reg * W
    
    
   # margins = margins*np.ones(loss.shape)
    #dmargins = np.zeros(margins.shape)
    #dmargins[margins>0] = 1
   # margins[np.arange(num_train),y] = -(np.sum(margins,axis=1)-1)
   # dW = X.T.dot(margins) / float(num_train)
   # dW += reg * W

    
    r_i=np.sum(margins>0,axis=1)
    R=np.array(margins>0,dtype=float) 
    R[np.arange(R.shape[0]),y]=-1*r_i
    dW=X.T.dot(R)
    dW /=X.shape[0]
    dW+=reg*W
    #Bool = Bool*np.ones(Loss.shape)
    #Bool[[y,np.arange(num_train)]] = -(np.sum(Bool,axis=0)-1)
    #dW = Bool.dot(X.T) / float(num_train)
    #dW += reg * W
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

    return loss, dW
