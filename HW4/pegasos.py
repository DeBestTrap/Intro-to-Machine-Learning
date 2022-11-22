'''Docstring for the pegasos.py module

This module implements the Pegasos algorithm from the paper:
“Pegasos: Primal Estimated sub-GrAdient SOlver for SVM,” by
S. Shalev-Shawtrz, Y. Singer, and N. Srebro.

To use: just call the `pegasos` function with specified parameters.
The rest of the functions are helper functions for pegasos.

'''
import time
import numpy as np
import sklearn.utils as skl 

def pegasos(X_:np.ndarray, y_:np.ndarray, max_iter:int, minibatch_size:int, lambd:np.float64=2.0e-4):
  '''
  Run the Pegasos algorithm.

  Parameters
  ----------
  X_: `np.ndarray`
    A 2D array that has N rows representing datums and d columns representing features
    Size: `(N, d)`
  y_: `np.ndarray`
    A 1D array that has N columns representing the class for each datumn.
    Size: `(N)`
  max_iter: `int`
    The maximum iterations to run the pegasos algorithm for.
  minibatch_size: `int`
    The number of minibatches to sample at each iteration.
  lambd: `np.float64`, default= 2.0e-4
    The regularization parameter of the SVM.

  Returns
  -------
  w_: `np.ndarray`
    The final weights after running Pegasos for `max_iter` iterations.
    Size: `(d)`
  obj: `np.ndarray`
    An array of all the objective values at each iteration, including the initial objective value.
    Size: `(max_iter+1)`
  losses: `np.ndarray`
    An array of all the loss at each iteration, including the inital loss.
    Size: `(max_iter+1)`
  cputs: `np.ndarray`
    An array of the cpu time at each iteration starting from 0. The first index is always 0.
    Size: `(max_iter+1)`
  '''
  N = X_.shape[0]
  d = X_.shape[1]
  w_ = np.zeros(d)

  obj = np.zeros(max_iter+1)
  losses = np.zeros(max_iter+1)
  cputs = np.zeros(max_iter+1)
  obj[0] = primal_objective(X_, y_, w_, lambd)
  losses[0] = np.sum(loss_all(X_,y_,w_))/N
  t = time.time()

  for itr in range(1, max_iter+1, 1):
    learning_rate = 1 / (lambd*itr)
    new_X_, new_y_ = skl.resample(X_, y_, n_samples=minibatch_size)
    new_X_, new_y_ = filter_batch(new_X_, new_y_, w_)
    w_ = (w_ - learning_rate * (lambd*w_ - (new_y_@new_X_)/minibatch_size))
    w_ = (min(1, 1.0/np.sqrt(lambd)/np.linalg.norm(w_))) * w_

    obj[itr] = primal_objective(X_, y_, w_, lambd)
    losses[itr] = np.sum(loss_all(X_,y_,w_))/N
    cputs[itr] = time.time()-t

  return w_, obj, losses, cputs

def primal_objective(X_:np.ndarray, y_:np.ndarray, w_:np.ndarray, lambd:np.float64):
  '''
  Calculates the objective value with the current weights.

  Parameters
  ----------
  X_: `np.ndarray`
    A 2D array that has N rows representing datums and d columns representing features
    Size: `(N, d)`
  y_: `np.ndarray`
    A 1D array that has N columns representing the class for each datumn.
    Size: `(N)`
  w_: `np.ndarray`
    A 1D array that has d columns representing the weights for each feature.
    Size: `(d)`
  lambd: `np.float64`, default= 2.0e-4
    The regularization parameter of the SVM.

  Returns
  -------
  The calculated objective value: `np.float64` 
  '''
  N = X_.shape[0]
  return lambd/2*(np.linalg.norm(w_)**2)+ np.sum(loss_all(X_, y_, w_))/N

def loss(x_:np.ndarray, y:int, w_:np.ndarray):
  '''
  Calculates the loss for a specific datum with the current weights.

  Note:
  The loss cannot be negative, if it is negative, it is capped to 0.

  Parameters
  ----------
  x_: `np.ndarray`
    A 1D array that has d columns representing the features for one datumn.
    Size: `(d)`
  y: `int`
    The class of this datum.
  w_: `np.ndarray`
    A 1D array that has d columns representing the weights for each feature.
    Size: `(d)`

  Returns
  -------
  The calculated loss for this datum: `np.float64` 
  '''
  loss = 1 - y * (x_@w_)
  return cap_to_0(0, loss)

def loss_all(X_:np.ndarray, y_:np.ndarray, w_:np.ndarray):
  '''
  Calculates the loss for all datums with the current weights.

  Note:
  The loss cannot be negative, if it is negative, it is capped to 0.

  Parameters
  ----------
  X_: `np.ndarray`
    A 2D array that has N rows representing datums and d columns representing features
    Size: `(N, d)`
  y_: `np.ndarray`
    A 1D array that has N columns representing the class for each datumn.
    Size: `(N)`
  w_: `np.ndarray`
    A 1D array that has d columns representing the weights for each feature.
    Size: `(d)`

  Returns
  -------
  A 1D array of the loss for each datum: `np.ndarray` of `np.float64` 
  Size: `(N)`
  '''
  loss_:np.ndarray = 1 - y_ * (X_@w_)
  return np.vectorize(cap_to_0)(loss_)

def cap_to_0(x):
  '''
  Parameters
  ----------
  x: a number
  
  Returns
  -------
  max(0, x)
  '''
  return max(0, x)

def filter_batch(X_:np.ndarray, y_:np.ndarray, w_:np.ndarray):
  '''
  Filters and shuffles datums that have a loss less than 0.

  Parameters
  ----------
  X_: `np.ndarray`
    A 2D array that has N rows representing datums and d columns representing features
    Size: `(N, d)`
  y_: `np.ndarray`
    A 1D array that has N columns representing the class for each datumn.
    Size: `(N)`
  w_: `np.ndarray`
    A 1D array that has d columns representing the weights for each feature.
    Size: `(d)`

  Returns
  -------
  Let `a` be the amount of datums that were removed.

  X_: `np.ndarray`
    A 2D array that has N rows representing datums and d columns representing features
    Size: `(N-a, d)`
  y_: `np.ndarray`
    A 1D array that has N-a columns representing the class for each datumn.
    Size: `(N-a)`
  '''
  loss_ = loss_all(X_, y_, w_)
  new_X_, new_y_ = skl.shuffle(X_[loss_ > 0], y_[loss_ > 0])
  return new_X_, new_y_
