import numpy as np


def Softmax_Loss(scores, y):
    '''
    Softmax Loss functions

    Args:
        scores: N by C matrix where N number of training examples and C number of classes
        y: Vector of trainig labels

    Returns: Softmax unregularised loss

    '''
    N = scores.shape[0]
    correct_scores = scores[np.arange(N), y]
    prob_sum = np.sum(np.exp(scores), axis=1)
    loss = np.exp(correct_scores) / prob_sum
    return np.sum(-np.log(loss)) / N