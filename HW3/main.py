from nb import *
from logistic_reg import *
from misc import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random


class ML:
    def __init__(self, filename: str, seed: int = 1):
        '''
        Parse the data from the file and create 6 ndarrays.
        It is split into original data, 80% training data, and 20% testing data.

        Three ndarrays represent the X_ feature matrix, the other three are the y values for those matricies.
        
        Parameters
        ----------
        filename: `str`
            the file name of the csv or data file
        seed: `int` (optional)
            default = `1`
        '''
        random.seed(seed)
        self.X_, self.y_ = ML.generate_data(filename)
        self.training_X_, self.training_y_, self.testing_X_, self.testing_y_ = ML.split_data(
            self.X_, self.y_, 80)

    def begin(self, num_splits: int, training_vector, type: str):
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
        type: `str`
            "logistic" for logistic regression        
            "NB" for Naive Bayers 
        '''
        for p in training_vector:
            total_error = []
            for i in range(num_splits):
                batch_size = (self.training_X_.shape[0]*p) // 100
                batch_X_, batch_y_, unused, unused = ML.split_data(
                    self.training_X_, self.training_y_, p)
                test_y_ = []
                if type == "logistic":
                    test_y_ = self.train_logistic_and_test(
                        batch_size, batch_X_, batch_y_)
                elif type == "NB":
                    mu0_, mu1_, var0_, var1_, phi1_ = find_gaussian_params(
                        batch_X_, batch_y_)
                    test_y_ = [NB_prediction(
                        x_, mu0_, mu1_, var0_, var1_, phi1_) for x_ in self.testing_X_]
                else:
                    exit(1)
                error = 0
                for j in range(len(test_y_)):
                    if test_y_[j] != self.testing_y_[j]:
                        error += 1

                error = error / self.testing_X_.shape[0]
                total_error.append(error)
            mean_error = sum(total_error) / num_splits
            var_error = sum([np.power(e - mean_error, 2)
                             for e in total_error]) / num_splits
            plt.errorbar(p, mean_error, yerr=var_error, fmt=".k")
            print(f"\nmean {mean_error}  |  var {var_error}")
        plt.ylabel("Error")
        plt.xlabel("Percent of training data")
        plt.show()

    def train_logistic_and_test(self, batch_size: int, batch_X_:np.ndarray, batch_y_:np.ndarray):
        '''
        Train the logistic regression at these parameters:
            learning rate = 0.01
            iterations = 350
            reg. constant = 0
        and return the results on the testing data set.
        
        Parameters
        ----------
        batch_size: `int`
            The number of rows in matrix X_
        batch_X_: 2D `np.ndarray`
            batch_size x d sized matrix containing features
        batch_y_: 1D `np.ndarray`
            d sized vector containing values
            
        Returns
        -------
        `List[int]` a list of 0 and 1s for the results of the
        testing data set.
        '''
        learning_rate = 0.01
        theta_, times = gradient_decent(
            X_=batch_X_,
            y_=batch_y_,
            iterations=350,
            learning_rate=learning_rate,
            batch_size=batch_size,
            function="logistic",
            lambd=0
        )
        test_y_ = []
        for i in range(self.testing_X_.shape[0]):
            if np.inner(theta_, np.append([1], self.testing_X_[i])) < 0:
                test_y_.append(0)
            else:
                test_y_.append(1)
        return test_y_

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

    def split_data(X_: np.ndarray, y_: np.ndarray, training_set_percent: int, make_training_set: bool = True):
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
        training_set_size = round(X_.shape[0] * training_set_percent / 100)
        testing_set_size = X_.shape[0] - training_set_size

        training_X_ = X_.copy()
        training_y_ = y_.copy()
        testing_X_ = np.zeros(shape=(testing_set_size, X_.shape[1]))
        testing_y_ = np.zeros(shape=testing_set_size)

        i = 0
        while(training_X_.shape[0] > training_set_size):
            idx = random.randrange(training_X_.shape[0])
            testing_X_[i] = training_X_[idx]
            testing_y_[i] = training_y_[idx]
            training_X_ = np.delete(training_X_, idx, 0)
            training_y_ = np.delete(training_y_, idx, 0)
            i += 1

        return [training_X_, training_y_, testing_X_, testing_y_]

def logisticRegression(filename:str, num_splits:int, train_percent):
    ml = ML(filename=filename, seed=random.randint(1, 100000))
    ml.begin(num_splits=num_splits, training_vector=train_percent, type="logistic")

def naiveBayesGaussian(filename:str, num_splits:int, train_percent):
    ml = ML(filename=filename, seed=random.randint(1, 100000))
    ml.begin(num_splits=num_splits, training_vector=train_percent, type="NB")

def runDefault():
    filename = "spambase.data"
    num_splits = 100
    train_percent = [5, 10, 15, 20, 25, 30]

    logisticRegression(filename, num_splits, train_percent)
    naiveBayesGaussian(filename, num_splits, train_percent)

if __name__ == "__main__":
    filename = input("Filename?\n")
    num_splits = int(input("num_splits? (int)\n"))
    train_percent_str = input('train_percent? (in the form of "x,x,x")\n')
    train_percent = [int(p) for p in train_percent_str.split(",")]

    ml = ML(filename=filename, seed=random.randint(1, 100000))
    ml.begin(num_splits=num_splits, training_vector=train_percent, type="logistic")
    ml.begin(num_splits=num_splits, training_vector=train_percent, type="NB")
