import numpy as np

def append_to_one(X_: np.ndarray):
    return np.append(np.array([1], dtype=X_.dtype), X_)