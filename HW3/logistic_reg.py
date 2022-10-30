import random
import time
from misc import *
import math
import numpy as np

def sigmoid_function(theta_, x_) -> float:
    '''
    Normalized sigmoid to avoid interger overflow.
    https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    '''
    z = np.dot(theta_.T, x_)
    if z >= 0:
        e = np.exp(-z)
        return 1.0/(1.0 + e)
    else:
        e = np.exp(z)
    return e / (1.0 + e)


def gradient_decent(
    X_: np.ndarray,
    y_: np.ndarray,
    iterations: int,
    learning_rate: float,
    batch_size: int,
    function: str,
    lambd: float
):
    N = X_.shape[0]
    function_types = {"logistic": sigmoid_function}
    f = function_types[function]
    theta_ = np.array([0.0 for i in range(X_.shape[1]+1)])
    times = []
    for i in range(iterations):
        start = time.time()
        theta_ = mini_batch_gradient_decent(
            theta_, X_, y_, N, learning_rate, batch_size, f, lambd)
        end = time.time()
        times.append(end - start)
    return (theta_, times)

def logistic_loss_all(theta_, X_, y_, N, lambd) -> float:
    sum = 0
    for i in range(N):
        reg = 0
        for theta in theta_:
            reg += theta**2
        new_x_ = np.append([1], X_[i])
        sum = sum - (
            y_[i] * math.log(sigmoid_function(theta_, new_x_)+1e9) +
            (1-y_[i]) * math.log(1-sigmoid_function(theta_, new_x_)+1e9)
        ) + (lambd * reg)
    return sum / N

def logistic_loss(theta_, x_, y, lambd) -> float:
    new_x_ = append_to_one(x_)
    a = y * math.log(sigmoid_function(theta_, new_x_)+1e9) + \
        (1-y) * math.log(1-sigmoid_function(theta_, new_x_)+1e9)
    return a

def find_graident(theta_, x_, y, function, lambd):
    new_x_ = np.append([1], x_)
    z = (function(theta_, new_x_)-y) * new_x_ + (lambd * theta_)
    return z


def mini_batch_gradient_decent(
    theta_: np.ndarray,
    X_: np.ndarray,
    y_: np.ndarray,
    N: int,
    learning_rate: float,
    batch_size: int,
    function,
    lambd: float
):
    if (batch_size != X_.shape[0]):
        batch = random.sample(range(N), k=batch_size)
        gradient = sum([find_graident(theta_, X_[i], y_[i], function, lambd)
                        for i in batch])
    else:
        gradient = sum([find_graident(theta_, X_[i], y_[i], function, lambd)
                        for i in range(X_.shape[0])])

    new_theta_ = theta_ - ((learning_rate * gradient)/batch_size)
    return new_theta_

def gradient_decent(
    X_: np.ndarray,
    y_: np.ndarray,
    iterations: int,
    learning_rate: float,
    batch_size: int,
    function: str,
    lambd: float
):
    N = X_.shape[0]
    function_types = {"logistic": sigmoid_function}
    f = function_types[function]
    theta_ = np.array([0.0 for i in range(X_.shape[1]+1)])
    times = []
    for i in range(iterations):
        start = time.time()
        theta_ = mini_batch_gradient_decent(
            theta_, X_, y_, N, learning_rate, batch_size, f, lambd)
        end = time.time()
        times.append(end - start)
    return (theta_, times)

