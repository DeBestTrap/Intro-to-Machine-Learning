See HW4.pdf for rendered LaTeX equations. Github does not render them in markdown files.

# Q1
## a)
### Symmetric 
$$
  K(x_i, x_j)
  =
  \sum^{m}_{a=1}
  w_a
  K_a(x_i,x_j)
$$
$$
\text{Since $K_1, ..., K_a$ are valid kernel functions:}
$$
$$
  K(x_i, x_j)
  =
  \sum^{m}_{a=1}
  w_a
  K_a(x_j,x_i)
  \triangleq
  K(x_j, x_i)
$$
$$
  \therefore 
  \text{K is symmetric}
$$
### Positive Semi-Definite
$$
\text{Using Mercer's Theorem, a Kernel is positive semi-definite if:}
$$
$$
\underline{c}^T\underline{K}\underline{c} \geq 0
$$
$$
  \sum^{n}_{i=1}
  \sum^{n}_{j=1}
  c_i
  c_j
  K(x_i, x_j)
  \geq
  0
$$
$$
  \sum^{n}_{i=1}
  \sum^{n}_{j=1}
  c_i
  c_j
  \sum^{m}_{a=1}
  w_a
  K_a(x_i,x_j)
  \geq
  0
$$
$$
  \sum^{m}_{a=1}
  w_a
  \sum^{n}_{i=1}
  \sum^{n}_{j=1}
  c_i
  c_j
  \langle
  \underline{\phi}(x_i),
  \underline{\phi}(x_j)
  \rangle
  \geq
  0
$$
$$
  \sum^{m}_{a=1}
  w_a
  \sum^{n}_{i=1}
  \sum^{n}_{j=1}
  c_i
  c_j
  \sum^{p}_{k=1}
  \phi_k(x_i)
  \phi_k(x_j)
  \geq
  0
$$
$$
  \sum^{m}_{a=1}
  w_a
  \sum^{p}_{k=1}
  \sum^{n}_{i=1}
  \sum^{n}_{j=1}
  c_i
  \phi_k(x_i)
  c_j
  \phi_k(x_j)
  \geq
  0
$$
$$
  \sum^{m}_{a=1}
  w_a
  \sum^{p}_{k=1}
  \left(
    \sum^{n}_{i=1}
    c_i
    \phi_k(x_i)
  \right)^2
  \geq
  0
$$
$$
\text{Since } w_i \geq 0, \forall w_i, \text{ The inequality holds, } \therefore \text{ K is positive semi-definite.}
$$

The function is symmetric and positive semi-definite, $\therefore K$ is a Kernel.

## b)
### Symmetric

$$
  K(x_i, x_j)
  =
  K_1(x_i,x_j)
  K_2(x_i,x_j)
$$
$$
\text{Since $K_1$ and $K_2$ are valid kernel functions:}
$$
$$
  K(x_i, x_j)
  =
  K_1(x_j,x_i)
  K_2(x_j,x_i)
  \triangleq
  K(x_j, x_i)
$$
$$
  \therefore 
  \text{K is symmetric}
$$

### Positive Semi-Definite
$$
\text{Using Mercer's Theorem, a Kernel is positive semi-definite if:}
$$
$$
\underline{c}^T\underline{K}\underline{c} \geq 0
$$
$$
  \sum^{n}_{i=1}
  \sum^{n}_{j=1}
  c_i
  c_j
  K(x_i, x_j)
  \geq
  0
$$
$$
  \sum^{n}_{i=1}
  \sum^{n}_{j=1}
  c_i
  c_j
  K_1(x_i, x_j)
  K_2(x_i, x_j)
  \geq
  0
$$
$$
  \sum^{n}_{i=1}
  \sum^{n}_{j=1}
  c_i
  c_j
  \langle
  \underline{\phi_1}(x_i),
  \underline{\phi_1}(x_j)
  \rangle
  \langle
  \underline{\phi_2}(x_i),
  \underline{\phi_2}(x_j)
  \rangle
  \geq
  0
$$
$$
  \sum^{n}_{i=1}
  \sum^{n}_{j=1}
  \sum^{p}_{k=1}
  c_i
  c_j
  \phi_{(1,k)}(x_i)
  \phi_{(1,k)}(x_j)
  \phi_{(2,k)}(x_i)
  \phi_{(2,k)}(x_j)
  \geq
  0
$$
$$
  \sum^{n}_{i=1}
  \sum^{n}_{j=1}
  \sum^{p}_{k=1}
  c_i
  \phi_{(1,k)}(x_i)
  \phi_{(2,k)}(x_i)
  c_j
  \phi_{(1,k)}(x_j)
  \phi_{(2,k)}(x_j)
  \geq
  0
$$
$$
  \sum^{p}_{k=1}
  \left(
    \sum^{n}_{i=1}
    c_i
    \phi_{(1,k)}(x_i)
    \phi_{(2,k)}(x_i)
  \right)^2
  \geq
  0
$$
$$
\text{Inequality holds, } \therefore \text{ K is positive semi-definite.}
$$

The function is symmetric and positive semi-definite, $\therefore K$ is a Kernel.

## c)
### Symmetric

$$
  K(x, x')
  =
  (xx' + 1)^{2015}
$$
$$
  K(x, x')
  =
  (x'x + 1)^{2015}
  \triangleq
  K(x', x)
$$
$$
  \therefore 
  \text{K is symmetric}
$$

### Positive Semi-Definite
$$
  K(x, x')
  =
  (xx' + 1)^{2015}
$$
$$
  K(x, x')
  =
  \sum^{2015}_{i=1}
  \left(
    \frac{2015!}
    {i!(2015-i)!}
    (x)^{i}
    (x')^{i}
  \right)
$$
$$
\text{Using Mercer's Theorem, a Kernel is positive semi-definite if:}
$$
$$
\underline{c}^T\underline{K}\underline{c} \geq 0
$$
$$
  \sum^{n}_{i=1}
  \sum^{n}_{j=1}
  c_i
  c_j
  K(x_i, x_j)
  \geq
  0
$$
$$
  \sum^{n}_{i=1}
  \sum^{n}_{j=1}
  c_i
  c_j
  \sum^{2015}_{k=1}
  \left(
    \frac{2015!}
    {k!(2015-k)!}
    (x_i)^{k}
    (x_j)^{k}
  \right)
  \geq
  0
$$
$$
  \sum^{2015}_{k=1}
  \frac{2015!}
  {k!(2015-k)!}
  \sum^{n}_{i=1}
  \sum^{n}_{j=1}
  c_i
  c_j
  (x_i)^{k}
  (x_j)^{k}
  \geq
  0
$$
$$
  \sum^{2015}_{k=1}
  \frac{2015!}
  {k!(2015-k)!}
  \left(
    \sum^{n}_{i=1}
    c_i
    (x_i)^{k}
  \right)^2
  \geq
  0
$$
$$
\text{Inequality holds, } \therefore \text{ K is positive semi-definite.}
$$

The function is symmetric and positive semi-definite, $\therefore K$ is a Kernel.
## d)
### Symmetric
$$
  K(x, x')
  =
  \exp
  \left(-
  \frac{(x-x')^2}
  {2}
  \right)
$$
$$
\text{Since $x-x'$ is being squared, it is also equal to:}
$$
$$
  K(x, x')
  =
  \exp
  \left(-
  \frac{(x'-x)^2}
  {2}
  \right)
  \triangleq
  K(x', x)
$$
$$
  \therefore 
  \text{K is symmetric}
$$
### Positive Semi-Definite
$$
  K(x, x')
  =
  \exp
  \left(-
  \frac{(x-x')^2}
  {2}
  \right)
$$
$$
  K(x, x')
  =
  \exp
  \left(-
  \frac{x^2-2xx'+(x')^2}
  {2}
  \right)
$$
$$
  K(x, x')
  =
  \exp
  \left(-
  \frac{x^2}{2}
  +
  xx'
  -
  \frac{(x')^2}{2}
  \right)
$$
$$
  K(x, x')
  =
  \frac{
    \exp
    \left(
    xx'
    \right)
  }{
    \exp
    \left(
    \frac{x^2}{2}
    \right)
    \exp
    \left(
    \frac{(x')^2}{2}
    \right)
  }
$$
$$
  K(x, x')
  =
  \frac{
    \sum_{k=0}^{\infty}
    \frac{(xx')^k}
    {k!}
  }{
    \exp
    \left(
    \frac{x^2}{2}
    \right)
    \exp
    \left(
    \frac{(x')^2}{2}
    \right)
  }
$$
$$
  K(x, x')
  =
  \sum_{k=0}^{\infty}
  \frac{(x)^k(x')^k}
  {
    k!
    \exp
    \left(
    \frac{x^2}{2}
    \right)
    \exp
    \left(
    \frac{(x')^2}{2}
    \right)
  }
$$
$$
\text{Using Mercer's Theorem, a Kernel is positive semi-definite if:}
$$
$$
\underline{c}^T\underline{K}\underline{c} \geq 0
$$
$$
  \sum^{n}_{i=1}
  \sum^{n}_{j=1}
  c_i
  c_j
  K(x_i, x_j)
  \geq
  0
$$
$$
  \sum^{n}_{i=1}
  \sum^{n}_{j=1}
  c_i
  c_j
  \sum_{k=0}^{\infty}
  \frac{(x_i)^k(x_j)^k}
  {
    k!
    \exp
    \left(
    \frac{x_i^2}{2}
    \right)
    \exp
    \left(
    \frac{x_j^2}{2}
    \right)
  }
  \geq
  0
$$
$$
  \sum_{k=0}^{\infty}
  \frac{1}{k!}
  \sum^{n}_{i=1}
  \sum^{n}_{j=1}
  c_i
  c_j
  \frac{(x_i)^k}
  {
    \exp
    \left(
    \frac{x_i^2}{2}
    \right)
  }
  \frac{(x_j)^k}
  {
    \exp
    \left(
    \frac{x_j^2}{2}
    \right)
  }
  \geq
  0
$$
$$
  \sum_{k=0}^{\infty}
  \frac{1}{k!}
  \left(
  \sum^{n}_{i=1}
  c_i
  \frac{(x_i)^k}
  {
    \exp
    \left(
    \frac{x_i^2}{2}
    \right)
  }
  \right)^2
  \geq
  0
$$
$$
\text{Inequality holds, } \therefore \text{ K is positive semi-definite.}
$$

The function is symmetric and positive semi-definite, $\therefore K$ is a Kernel.
# Q2

## Summary and Results

max_iter = 100

lambda = 1e5 (A high regularization coefficient was used to stabilize the plots quickly)

![r](../images/part2_results.png)
![r](../images/part2_avgstd.png)

## Code
Code can be found on [Github](https://github.com/DeBestTrap/Intro-to-Machine-Learning/tree/main/HW4).

Pegasos algorithm:

https://github.com/DeBestTrap/Intro-to-Machine-Learning/blob/main/HW4/pegasos.py

Code to run and plot pegasos algorithm (for instructions to run code see [README.md](https://github.com/DeBestTrap/Intro-to-Machine-Learning/tree/main/HW4#readme)):

https://github.com/DeBestTrap/Intro-to-Machine-Learning/blob/main/HW4/mysgdsvm.py