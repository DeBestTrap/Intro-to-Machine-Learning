import statistics
from misc import *
import numpy as np

def find_gaussian_params(X_, y_):
    '''
    Finds the mean and variance when y=0 or y=1 when for a multivariate 
    gaussian distribution with the assumption of Naive Bayes.
    '''
    N = X_.shape[0]
    d = X_.shape[1]

    mu0_ = np.zeros(shape=d)
    mu1_ = np.zeros(shape=d)
    num1s = 0
    for n in range(N):
        if (y_[n] == 1.0):
            mu1_ += X_[n]
            num1s += 1
        else:
            mu0_ += X_[n]
    mu0_ /= (N-num1s)
    mu1_ /= num1s

    var0_ = np.zeros(shape=d)
    var1_ = np.zeros(shape=d)
    num1s = 0
    for n in range(N):
        if (y_[n] == 1.0):
            var1_ += np.power(X_[n]-mu1_, 2)
            num1s += 1
        else:
            var0_ += np.power(X_[n]-mu0_, 2)
    var0_ /= (N-num1s)
    var1_ /= num1s

    phi1_ = sum(y_)/N

    return [mu0_, mu1_, var0_, var1_, phi1_]


def pdf_gaussian_NB(x_, mu_, var_):
    '''
    Find the result of gaussian pdf assuming Naive Bayes
    $ \prod_{i=1}^{d} P(x_n^(i) | y_n) $
    '''
    d = x_.shape[0]
    prod = 1
    for i in range(d):
        if (var_[i] == 0.0):
            var = 0.00001
        else:
            var = var_[i]
        prod *= statistics.NormalDist(mu_[i], var).pdf(x_[i])
    return prod


def NB_prediction(x_, mu0_, mu1_, var0_, var1_, phi1_):
    '''
    Returns
    -------
    0 if the probabilty of X_ being class 0 is higher than prob. being 1
    1 if ^^^ 0 is lower ^^^ than 1
    '''
    class0_prob = pdf_gaussian_NB(x_, mu0_, var0_) * (1-phi1_)
    class1_prob = pdf_gaussian_NB(x_, mu1_, var1_) * phi1_
    if (class0_prob > class1_prob):
        return 0
    else:
        return 1
