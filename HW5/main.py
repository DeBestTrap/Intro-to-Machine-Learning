import random
import numpy as np
import pandas as pd
import sklearn.utils as skutil
import matplotlib.pyplot as plt
from my_neural_network import MyNeuralNetwork
from my_neural_network import Data
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

class ML:
    def __init__(self, filename: str, seed: int = 1, optim_type: str = "SGD"):
        '''
        Parse the data from the file and create 6 ndarrays.
        
        Parameters
        ----------
        filename: `str`
            the file name of the csv or data file
        seed: `int` (optional)
            default = `1`
        '''
        random.seed(seed)
        self.X_, self.y_ = ML.generate_data(filename)
        self.optim_type = optim_type

    def begin(self, num_splits: int, training_vector, nn_stack):
        '''
        Begin the training and testing of data depending on the type chosen.
        The error results of each split are averaged together and displayed on a graph

        Parameters
        ----------
        num_splits: `int`
            The number of runs to do for each percentage in training_vector.
            The runs are averaged togther at the end.
        training_vector: `List[int]`
            A list of percentages of the training data to train on.
            e.g. 50 for 50% of the training data.
        nn_stack: `nn.Sequential`
            the 
        '''
        test_errors = np.zeros((num_splits, len(training_vector)))

        for i in range(num_splits):
            self.training_X_, self.training_y_, self.testing_X_, self.testing_y_ = ML.split_data(
                self.X_, self.y_, 80)
            self.shuffle_training_data()
            for j, p in enumerate(training_vector):
                self.split_training_X_, self.split_training_y_, unused, unused = ML.split_data(
                    self.training_X_, self.training_y_, p)
                self.shuffle_split_training_data()
                
                my_NN = self.train_neural_network(nn_stack)

                error = self.test_neural_network(my_NN)
                
                test_errors[i][j] = error

        for j, percent in enumerate(training_vector):
            print(f"{percent}% training data: mean error = {np.mean(test_errors[:,j]):.04f}, std. deviation = {np.std(test_errors[:,j]):.04f}")
        return test_errors, np.mean(test_errors, axis=0), np.std(test_errors,axis=0)

    def train_neural_network(self, nn_stack):
        '''
        trains the neural network and returns the final neural network with the finished weights

        nn_stack: the stack of operations that the neural network uses
        '''
        data = Data(self.split_training_X_, self.split_training_y_, device)
        loader = DataLoader(dataset=data,batch_size=int(self.split_training_X_.shape[0]))
        my_NN = MyNeuralNetwork(nn_stack).to(device)

        epochs = 50
        learning_rate = 1e-1
        loss_fn = nn.BCELoss()
        if (self.optim_type == "Adam"):
            optimizer=torch.optim.Adam(my_NN.parameters(), lr=learning_rate)
        optimizer = torch.optim.SGD(my_NN.parameters(), lr=learning_rate)

        loss_average_list = []
        loss_list = []
        loss_list_all = []
        for t in range(epochs):
            for x_, y in loader:
                optimizer.zero_grad()
                output = my_NN(x_)
                loss = loss_fn(output, y.unsqueeze(-1))
                loss_list.append(loss.item())
                loss_list_all.append(loss.item())
                loss.backward()
                optimizer.step()
            loss_average_list.append(np.mean(loss_list))
            loss_list = []

        # Plot Loss
        # step = np.linspace(0,t,len(loss_list_all))
        # plt.plot(step, np.array(loss_list_all))
        # step = np.linspace(0,epochs,len(loss_average_list))
        # plt.plot(step, np.array(loss_average_list))
        # plt.show()
        return my_NN

    def test_neural_network(self, my_NN):
        '''
        take a neural network and tests the results with the testing set

        returns the error rate
        '''

        correct = 0
        total = 0
        data = Data(self.testing_X_, self.testing_y_, device)

        with torch.no_grad():
            for x_, y in data:
                output = my_NN(x_)
                y_hat = np.where(output < 0.5, 0, 1)
                total += 1
                correct += (y_hat == y.numpy()).sum().item()
        error = 100 * (1- correct / total)
        # print(error)
        return error

    def generate_data(filename: str):
        '''
        Given a filename to a file that has data stored in the form of a CSV,
        parse the file and generate a vector of feature vectors (X_) and the 
        correpsonding values (y_). The last colnum will be detached and used as y_.

        Parameters
        ----------
        filename: `str`
          The filename of the .data or .csv to be read.
          The rows should be in the form v1,v2,...,vn.

        Returns
        -------
        `Tuple(np.ndarray, np.ndarray)` where the first value is the X_ feature
        matrix and the second value is the y_ value vector
        '''
        sheet = pd.read_csv(filename, header=None)
        X_ = sheet.to_numpy()
        y_ = X_[:, -1]
        X_ = np.delete(X_, np.s_[-1:], axis=1)
        return (X_, y_)

    def split_data(X_: np.ndarray, y_: np.ndarray, training_set_percent: int, make_testing_set: bool = True):
        '''
        Split the data into (training_set_percent)% training data and (100-training_set_percent)% testing data

        Parameters
        ----------
        X_: `np.ndarray`
          The X matrix of data where each row is a feature vector.
        y_: `np.ndarray`
          The y vector of values where each corresponds to the feature vector in X.

        Returns
        ------
        `List[np.ndarray]`
          This list contains 4 np.ndarrays, in the form of
          `[training_X_, training_y_, testing_X_, testing_y_]`
        '''
        X_0 = X_[y_ == 0]
        X_1 = X_[y_ == 1]
        y_0 = y_[y_ == 0]
        y_1 = y_[y_ == 1]

        split_idx_0 = round(X_0.shape[0] * training_set_percent * 0.01)
        split_idx_1 = round(X_1.shape[0] * training_set_percent * 0.01)

        X_0, y_0 = skutil.shuffle(X_0, y_0)
        X_1, y_1 = skutil.shuffle(X_1, y_1)

        training_X_ = np.concatenate((X_0[:split_idx_0], X_1[:split_idx_1]), axis=0)
        training_y_ = np.concatenate((y_0[:split_idx_0], y_1[:split_idx_1]), axis=0)

        if make_testing_set:
            testing_X_ = np.concatenate((X_0[split_idx_0:], X_1[split_idx_1:]), axis=0)
            testing_y_ = np.concatenate((y_0[split_idx_0:], y_1[split_idx_1:]), axis=0)
        else:
            testing_X_ = np.empty(1)
            testing_y_ = np.empty(1)

        return [training_X_, training_y_, testing_X_, testing_y_]
    
    def shuffle_training_data(self):
        '''
        shuffles the training data
        '''
        self.training_X_, self.training_y_ = skutil.shuffle(self.training_X_, self.training_y_)

    def shuffle_split_training_data(self):
        '''
        shuffles the split training data
        '''
        self.split_training_X_, self.split_training_y_ = skutil.shuffle(self.split_training_X_, self.split_training_y_)

    def plot_results(train_percent, mean, std, title):
        '''
        Plot the results and save it to images folder.

        mean: array of means
        std: array of std
        title: the title of the plot
        '''
        plt.style.use('ggplot')
        plt.rc('axes', labelsize=20)
        plt.rc('xtick', labelsize=20)
        plt.rc('ytick', labelsize=20)
        plt.rc('legend', fontsize=18)
        plt.rc('lines', linewidth=4)
        fig = plt.figure(figsize=(8,6))
        plt.errorbar(train_percent, mean, yerr=std, fmt='ro--', capsize=10, elinewidth=4)
        plt.xlabel('training %', fontsize=24)
        plt.ylabel('test error rate', fontsize=24)
        plt.title(f'{title}', fontsize=24)
        # plt.show()
        plt.savefig(f'./images/{title.replace(" ", "_")}.png')

num_splits = 100
training_vector = [10, 20, 30]
# ml = ML("test.data")
ml = ML("spambase.data")
input_layers = ml.X_.shape[1]
output_layers = 1 

nn_stack = nn.Sequential(
    nn.Linear(input_layers, output_layers),
    nn.Sigmoid(),
)
errors, mean, stdev = ml.begin(num_splits, training_vector, nn_stack)
ML.plot_results([10,20,30], mean, stdev, f"Logisitic Regression {ml.optim_type}")

nn_stack = nn.Sequential(
    nn.Linear(input_layers, 10),
    nn.ReLU(),
    nn.Linear(10, output_layers),
    nn.Sigmoid(),
)
errors, mean, stdev = ml.begin(num_splits, training_vector, nn_stack)
ML.plot_results([10,20,30], mean, stdev, f"X-10-1 Relu {ml.optim_type}")

nn_stack = nn.Sequential(
    nn.Linear(input_layers, 10),
    nn.Tanh(),
    nn.Linear(10, output_layers),
    nn.Sigmoid(),
)
errors, mean, stdev = ml.begin(num_splits, training_vector, nn_stack)
ML.plot_results([10,20,30], mean, stdev, f"X-10-1 Tanh {ml.optim_type}")

nn_stack = nn.Sequential(
    nn.Linear(input_layers, 10),
    nn.ReLU(),
    nn.Linear(10, 10),
    nn.ReLU(),
    nn.Linear(10, output_layers),
    nn.Sigmoid(),
)
errors, mean, stdev = ml.begin(num_splits, training_vector, nn_stack)
ML.plot_results([10,20,30], mean, stdev, f"X-10-10-1 Relu {ml.optim_type}")

nn_stack = nn.Sequential(
    nn.Linear(input_layers, 30),
    nn.ReLU(),
    nn.Linear(30, output_layers),
    nn.Sigmoid(),
)
errors, mean, stdev = ml.begin(num_splits, training_vector, nn_stack)
ML.plot_results([10,20,30], mean, stdev, f"X-30-1 Relu {ml.optim_type}")


ml.optim_type = "Adam"

nn_stack = nn.Sequential(
    nn.Linear(input_layers, output_layers),
    nn.Sigmoid(),
)
errors, mean, stdev = ml.begin(num_splits, training_vector, nn_stack)
ML.plot_results([10,20,30], mean, stdev, f"Logisitic Regression {ml.optim_type}")

nn_stack = nn.Sequential(
    nn.Linear(input_layers, 10),
    nn.ReLU(),
    nn.Linear(10, output_layers),
    nn.Sigmoid(),
)
errors, mean, stdev = ml.begin(num_splits, training_vector, nn_stack)
ML.plot_results([10,20,30], mean, stdev, f"X-10-1 Relu {ml.optim_type}")

nn_stack = nn.Sequential(
    nn.Linear(input_layers, 10),
    nn.Tanh(),
    nn.Linear(10, output_layers),
    nn.Sigmoid(),
)
errors, mean, stdev = ml.begin(num_splits, training_vector, nn_stack)
ML.plot_results([10,20,30], mean, stdev, f"X-10-1 Tanh {ml.optim_type}")

nn_stack = nn.Sequential(
    nn.Linear(input_layers, 10),
    nn.ReLU(),
    nn.Linear(10, 10),
    nn.ReLU(),
    nn.Linear(10, output_layers),
    nn.Sigmoid(),
)
errors, mean, stdev = ml.begin(num_splits, training_vector, nn_stack)
ML.plot_results([10,20,30], mean, stdev, f"X-10-10-1 Relu {ml.optim_type}")

nn_stack = nn.Sequential(
    nn.Linear(input_layers, 30),
    nn.ReLU(),
    nn.Linear(30, output_layers),
    nn.Sigmoid(),
)
errors, mean, stdev = ml.begin(num_splits, training_vector, nn_stack)
ML.plot_results([10,20,30], mean, stdev, f"X-30-1 Relu {ml.optim_type}")