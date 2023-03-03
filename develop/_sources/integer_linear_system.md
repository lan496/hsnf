# Integer Linear system

## Integer linear system

For given {math}`\mathbf{A} \in \mathbb{Z}^{m \times n}` and {math}`\mathbf{b} \in \mathbb{Z}^{m}`, consider to solve integer linear system {math}`\mathbf{Ax} = \mathbf{b}` in {math}`\mathbf{x} \in \mathbb{Z}^{n}`.
Let the Hermite normal form of {math}`\mathbf{A}` be {math}`\mathbf{H} = \mathbf{AR}`, where {math}`\mathbf{R}` is unimodular and {math}`\mathbf{H}` is lower triangular.
The given linear system is 
```{math}
  \begin{pmatrix}
    H_{11} &        & \mathbf{O} & \vdots     \\
    \vdots & \ddots &            & \mathbf{0} \\
    H_{r1} & \ldots & H_{rr}     & \vdots     \\
    \ldots & \mathbf{0} & \ldots & \mathbf{O} \\
  \end{pmatrix}
  \mathbf{y} = \mathbf{b}
```
where {math}`\mathbf{y} := \mathbf{R}^{-1}\mathbf{x}`.
A special solution, {math}`\mathbf{x}_{\mathrm{special}} = \mathbf{R}\mathbf{y}_{\mathrm{special}}`, is determined by Gaussian elimination if exists.
A general solution for {math}`\mathbf{Hy}=\mathbf{0}` is given by
```{math}
  \mathbf{y}
    &= \begin{pmatrix} 0 \\ \vdots \\ 0 \\ n_{r+1} \\ \vdots \\ n_{m} \end{pmatrix}
        \quad (\forall n_{r+1},\cdots, n_{m} \in \mathbb{Z}).
```

## Frobenius congruent

For given {math}`\mathbf{A} \in \mathbb{Z}^{m \times n}` and {math}`\mathbf{b} \in \mathbb{Z}^{m}`, consider to solve Frobenius congruent {math}`\mathbf{Ax} \equiv \mathbf{b} \, (\mathrm{mod}\, \mathbb{R}/\mathbb{Z})` for {math}`\mathbf{x} \in \mathbb{R}^{n}`.
Let the Smith normal form of {math}`\mathbf{A}` be {math}`\mathbf{D} = \mathbf{LAR}`, where {math}`\mathbf{L}` and {math}`\mathbf{R}` are unimodular matrices.
```{math}
  \mathbf{LAx} &= \mathbf{Lb} + \mathbb{Z}^{n} \\
  \mathbf{Dy}  &= \mathbf{v} + \mathbb{Z}^{n} \quad \mbox{where}\, \mathbf{y}:= \mathbf{R}^{-1} \mathbf{x},\, \mathbf{v}:= \mathbf{Lb} \\
  \mathbf{y}
    &= \begin{pmatrix} \frac{v_{1}}{D_{11}} \\ \vdots \\ \frac{v_{r}}{D_{rr}} \\ 0 \\ \vdots \\ 0 \end{pmatrix}
        + \begin{pmatrix} \frac{1}{D_{11}} n_{1} \\ \vdots \\ \frac{1}{D_{rr}} n_{r} \\ 0 \\ \vdots \\ 0 \end{pmatrix}
        + \begin{pmatrix} 0 \\ \vdots \\ 0 \\ a_{r+1} \\ \vdots \\ a_{m} \end{pmatrix}
        \quad (\forall n_{1}, \cdots, n_{r} \in \mathbb{Z}, \forall a_{r+1}, \cdots, a_{m} \in \mathbb{R})
```

## Modular integer linear system

For given {math}`\mathbf{A} \in \mathbb{Z}^{m \times n}` and {math}`\mathbf{b} \in \mathbb{Z}^{m}`, consider to solve modular integer linear system {math}`\mathbf{Ax} \equiv \mathbf{b} \, (\mathrm{mod} \, q)` in {math}`\mathbf{x} \in \mathbb{Z}^{n}`.

First, We consider finding one of solutions of {math}`\mathbf{Ax}=\mathbf{b} \, (\mathrm{mod} p^{k})` for given {math}`\mathbf{A} \in \mathbb{Z}^{m \times n}` and {math}`\mathbf{b} \in \mathbb{Z}^{m}`.

Let the Smith normal form of {math}`\mathbf{A}` be {math}`\mathbf{D} = \mathbf{LAR}`, where {math}`\mathbf{L}` and {math}`\mathbf{R}` are unimodular matrices.
Now it is sufficient to solve {math}`\mathbf{Dy} = \mathbf{Lb} \, (\mathrm{mod} p^{k})` for {math}`\mathbf{y} \in \mathbb{Z}^{m}`.
After we obtain {math}`\mathbf{y}`, we easily obtain the solution of the original linear systems as {math}`\mathbf{x} = \mathbf{Ry} \, (\mathrm{mod} p^{k})`.

We write the SNF as {math}`\mathbf{D} = \mathrm{diag}(d_{1}, \cdots, d_{r}, 0, \cdots)`.
Then, the {math}`i`th element of {math}`\mathbf{y}` is determined by solving
```{math}
d_{i} y_{i} = [\mathbf{Lb}]_{i} \quad (\mathrm{mod} \, p^{k})
```
for {math}`i=1, \cdots, r`.
The above can be solved by the extended Euclidean algorithm.
If {math}`\mathrm{GCD}(d_{i}, p^{k})` does not divide {math}`p^{k}` for some {math}`i`, no solution exists.

Note that multiple solutions {math}`\mathbf{x}` may exist for {math}`k \geq 2`.
If we need all the solutions, the next task is to construct an integer lattice formed by {math}`\mathbf{Ax}=\mathbf{0} \, (\mathrm{mod}\,p^{k})`.
This problem can also be solved by a lattice algorithm shown in [this lecture note](https://cseweb.ucsd.edu/classes/wi12/cse206A-a/lec4.pdf).

The generalization from prime powers {math}`p^{k}` to other composite numbers {math}`l` is straightforward.
After factoring {math}`l` and solving the linear equations for each prime power, we can get a solution of {math}`\mathbf{Ax} = \mathbf{b} \, (\mathrm{mod}\, n)` by the Chinese remainder theorem.
