import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W).T

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################


  #Loss function http://cs231n.github.io/linear-classify/

  num_train = X.shape[0]
  num_classes = W.shape[1]
  log_loss = 0
  #normalisation trick
  #scores =  (X.T - np.max(X, axis=1)).T.dot(W)
  for i in xrange(num_train):
    scores = X[i].dot(W)
    scores = (scores.T - np.max(scores)).T
    #print np.max(scores)
    correct_class_score = scores[y[i]]
    #print correct_class_score
    loss = np.exp(correct_class_score)/np.sum(np.exp(scores))
    #print loss

    for j in xrange(num_classes):
      if j == y[i]:
        dW[j] += X[i]*(loss - 1)
      else:
        dW[j] += np.exp(scores[j])/np.sum(np.exp(scores))*X[i]
    log_loss += -np.log(loss)

  loss = log_loss/num_train + 0.5 * reg * np.sum(W*W)

  dW /= num_train + reg*W.T
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW.T


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]
  log_loss = 0


  scores = X.dot(W)
  correct_class_scores = scores[np.arange(num_train), y]
  # N vector
  prob_sum = np.sum(np.exp(scores), axis=1)
  loss = np.exp(correct_class_scores)/prob_sum
  print scores.shape
  grad_helper = np.tile(loss.T, (num_classes, 1)).T

  print np.sum(np.exp(scores), keepdims=True ).shape
  print grad_helper.shape


  grad_helper[np.arange(num_train), y] -= 1



  loss = np.sum(-np.log(loss))/num_train + 0.5*reg*np.sum(W*W)
  dW = np.dot(X.T, grad_helper)/num_train + reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

