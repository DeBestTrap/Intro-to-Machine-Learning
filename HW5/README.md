# a) Summary of Methods and Results

It seems that adding more layers and changing the hidden layer functions dont seem to make a large difference in changing the error rate.

## i) Optimizer: SGD
<p float="left">
  <img src="./images/Logisitic_Regression_SGD.png" width="19%" />
  <img src="./images/X-10-1_Relu_SGD.png" width="19%" />
  <img src="./images/X-10-1_Tanh_SGD.png" width="19%" />
  <img src="./images/X-30-1_Relu_SGD.png" width="19%" />
  <img src="./images/X-10-10-1_Relu_SGD.png" width="19%" />
</p>

## ii) Optimizer: Adam
<p float="left">
  <img src="./images/Logisitic_Regression_Adam.png" width="19%" />
  <img src="./images/X-10-1_Relu_Adam.png" width="19%" />
  <img src="./images/X-10-1_Tanh_Adam.png" width="19%" />
  <img src="./images/X-30-1_Relu_Adam.png" width="19%" />
  <img src="./images/X-10-10-1_Relu_Adam.png" width="19%" />
</p>

# b) Code

The main file where datasets are generated and plots are made:

https://github.com/DeBestTrap/Intro-to-Machine-Learning/blob/main/HW5/main.py

The neural network class (the stack of operations is created in the main file and passed into the class):

https://github.com/DeBestTrap/Intro-to-Machine-Learning/blob/main/HW5/my_neural_network.py