This homework implements the Pegasos algorithm from [Pegasos: Primal Estimated sub-GrAdient SOlver for SVM](https://github.com/DeBestTrap/Intro-to-Machine-Learning/blob/main/HW4/files/pegasos.pdf) on the [MNIST-13](https://github.com/DeBestTrap/Intro-to-Machine-Learning/blob/main/HW4/MNIST-13.csv) dataset.

How to use:
```
python mysgdsvm.py filename k numruns

params
------
  filename: filename of the dataset
  k: the minibatch size
  numruns: number of runs

optional flags
--------------
  --plot: plots the data and shows it
  --results: run the same k's as in the "Summary and Results" section.
    When using this flag, the k argument is not used but still must 
    be included.
```

Example, shows the plot for the "Summary and Results" section:
```
python mysgdsvm.py "MNIST-13.csv" 1 5 --plot --results
```


# Q1
## a)
![a](./images/a.png)
## b)
![b](./images/b.png)
## c)
![c](./images/c.png)
## d)
![d](./images/d.png)

# Q2

## Summary and Results
```
max_iter = 100
lambda = 1e5
```
![r](./images/part2_results.png)
![r](./images/part2_avgstd.png)

## Code
Code and README can be found on Github.

