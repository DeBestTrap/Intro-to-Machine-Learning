import time
import numpy as np
import sklearn.utils as skl 

def pegasos(X_:np.ndarray, y_:np.ndarray, max_iter:int, minibatch_size:int, lambd:np.float64=2.0e-4):
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
    new_X_, new_y_ = fetch_batch(new_X_, new_y_, w_)
    w_ = (w_ - learning_rate * (lambd*w_ - (new_y_@new_X_)/minibatch_size))
    w_ = (min(1, 1.0/np.sqrt(lambd)/np.linalg.norm(w_))) * w_

    obj[itr] = primal_objective(X_, y_, w_, lambd)
    losses[itr] = np.sum(loss_all(X_,y_,w_))/N
    cputs[itr] = time.time()-t

  return w_, obj, losses, cputs


def primal_objective(X_:np.ndarray, y_:np.ndarray, w_:np.ndarray, lambd:np.float64):
  N = X_.shape[0]
  return lambd/2*(np.linalg.norm(w_)**2)+ np.sum(loss_all(X_, y_, w_))/N

def loss(x_:np.ndarray, y:int, w_:np.ndarray):
  loss = 1 - y * (x_@w_)
  return max(0, loss)

def loss_all(X_:np.ndarray, y_:np.ndarray, w_:np.ndarray):
  loss_:np.ndarray = 1 - y_ * (X_@w_)
  return np.vectorize(cap_to_0)(loss_)

def cap_to_0(x):
  return 0 if x < 0 else x

def fetch_batch(X_:np.ndarray, y_:np.ndarray, w_:np.ndarray):
  loss_ = loss_all(X_, y_, w_)
  new_X_, new_y_ = skl.shuffle(X_[loss_ > 0], y_[loss_ > 0])
  return new_X_, new_y_
