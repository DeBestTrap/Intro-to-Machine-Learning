import numpy as np
import pandas as pd
import pegasos as ps
import matplotlib.pyplot as plt
import sys

X_ = None
y_ = None

def get_data(filename:str):
  '''
  Retrieves data and uses the first column as class labels. 
  '''
  global X_
  global y_
  if X_ is None:
    df = pd.read_csv(filename, header=None)
    X_ = df.to_numpy()
    y_ = X_[:,0]
    X_ = np.delete(X_, 0, axis=1)
  return X_, y_

def process_y_(y_:np.ndarray):
  '''
  Process the class labels to be 1 and -1.
  '''
  new_y_ = np.zeros(y_.shape[0])
  for i, y in enumerate(y_):
    if y == 3:
      new_y_[i] = -1
    elif y == 1:
      new_y_[i] = 1
  return new_y_
  
def mysgdsvm(filename, k, numruns, max_iter=100, lambd=1e5):
  '''
  Run the Pegasos algorithm `numruns` times and report the average runtime
  and std.

  If the --plot flag was used then plot the data and show the figure.
  '''
  X_, y_ = get_data(filename)
  y_ = process_y_(y_)

  obj_ = []
  losses_ = []
  cputs_ = []
  for r in range(numruns):
    w_, obj, losses, cputs = ps.pegasos(X_, y_, max_iter=max_iter, minibatch_size=k, lambd=lambd)
    obj_.append(obj)
    losses_.append(losses)
    cputs_.append(cputs)
  print(f"Avg runtime for {numruns} runs with minibatch size of {k}: {np.average([cputs[-1] for cputs in cputs_]):.2f} sec")
  print(f"Std runtime for {numruns} runs with minibatch size of {k}: {np.std([cputs[-1] for cputs in cputs_]):.2f} sec")
  if "--plot" in sys.argv and "--results" not in sys.argv:
    plot_data_and_show(range(max_iter+1), obj_, [f"Run {r+1}" for r in range(numruns)], x_label="Iterations", y_label="Objective", title=f"k={k}")
  return obj_, losses_, cputs_

def results(filename, k, numruns, max_iter=100, lambd=1e5):
  '''
  Run the Pegasos algorithm `numruns` times for k values of `[1, 20, 100, 200, N]` and report the average runtime
  and std.

  The k parameter is not used.

  If the --plot flag was used then plot the data and show the figure.
  '''
  X_, y_ = get_data(filename)
  y_ = process_y_(y_)

  plot = False
  if "--plot" in sys.argv:
    plot = True
    x_label = "Iteration"
    y_label = "Objective"
    fig = plt.figure(figsize=(8*5, 4))
    fig.supxlabel(x_label, fontsize=24)
    fig.supylabel(y_label, fontsize=24)
    fig.suptitle(f'{y_label} vs. {x_label}', fontsize=24)

  for i, k in enumerate([1, 20, 100, 200, X_.shape[0]]):
    obj_, losses_, cputs_ = mysgdsvm(filename, k, numruns, max_iter, lambd)
    if plot:
      plot_all_data(range(max_iter+1), obj_, [f"Run {r+1}" for r in range(numruns)], x_label=x_label, y_label=y_label, title=f"k={k}", subplot=i+1)

  if plot:
    plt.show()

def plot_all_data(x, ys, labels, x_label=None, y_label=None, title=None, subplot=1):
  '''
  Plots the data in 5 subplots.
  Does not show the figure.
  '''
  ax = plt.subplot(1,5, subplot)
  for i in range(len(ys)):
    ax.plot(x, ys[i], label=labels[i])
  ratio = 1.0
  xleft, xright = ax.get_xlim()
  ybottom, ytop = ax.get_ylim()
  ax.set_aspect(abs((xright-xleft)/(ybottom-ytop))*ratio)
  ax.set_title(title)
  ax.legend()

def plot_data_and_show(x, ys, labels, x_label=None, y_label=None, title=None):
  '''
  Plots the data in a plot and show the figure.
  '''
  plt.figure(figsize=(8,6))
  for i in range(len(ys)):
    plt.plot(x, ys[i], label=labels[i])
  plt.legend()
  plt.xlabel(x_label, fontsize=24)
  plt.ylabel(y_label, fontsize=24)
  plt.title(f'{y_label} vs. {x_label}, '+title, fontsize=24)
  plt.show()

def main():
  filename = sys.argv[1]
  k = int(sys.argv[2])
  numruns = int(sys.argv[3])

  if "--results" in sys.argv:
    results(filename, k, numruns)
  else:
    mysgdsvm(filename, k, numruns)


if __name__ == "__main__":
  main()