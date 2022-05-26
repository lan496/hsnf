Definition
==========

Smith normal form
-----------------

Let :math:`\mathbf{M}` be :math:`m \times n` integer matrix.
There exist some unimodular matrices :math:`\mathbf{L} \in \mathbb{Z}^{m \times m}` and :math:`\mathbf{R} \in \mathbb{Z}^{n \times n}` such that

.. math::
    \mathbf{D}
    :=
    \mathbf{LMR}
    =
    \begin{pmatrix}
        d_{1}      &        & \mathbf{O}          & \mathbf{0} \\
                   & \ddots &                     & \vdots     \\
        \mathbf{O} &        & d_{r}               & \mathbf{0} \\
        \mathbf{0} & \cdots & \mathbf{0} & \mathbf{O}
    \end{pmatrix}

where :math:`d_{i}` is positive integer and :math:`d_{i+1}` devides :math:`d_{i}`.
Then :math:`\mathbf{D}` is called Smith normal form.

Row-style Hermite normal form
-----------------------------

Let :math:`\mathbf{M}` be :math:`m \times n` integer matrix.
It has a row-style Hermite normal form :math:`\mathbf{H}` if there exists a unimodular matrices :math:`\mathbf{L} \in \mathbb{Z}^{m \times m}` such that :math:`\mathbf{H}=\mathbf{LM}` satisfied the following conditions

1. :math:`H_{ij} \geq 0 \quad (1 \leq i \leq m, 1 \leq j \leq n)`
2. :math:`H_{ij} = 0 \quad (i > j \, \wedge i > r)`
3. :math:`H_{ij} < H_{jj} \quad (i < j, 1 \leq j \leq r)`
4. :math:`r = \mathrm{rank} \mathbf{A}`

If :math:`\mathbf{M}` is full rank, the Hermite normal form :math:`\mathbf{H}` is uniquely determined.

Column-style Hermite normal form
--------------------------------

Let :math:`\mathbf{M}` be :math:`m \times n` integer matrix.
It has a column-style Hermite normal form :math:`\mathbf{H}` if there exists a unimodular matrices :math:`\mathbf{R} \in \mathbb{Z}^{n \times n}` such that :math:`\mathbf{H}=\mathbf{MR}` satisfied the following conditions

1. :math:`H_{ij} \geq 0 \quad (1 \leq i \leq m, 1 \leq j \leq n)`
2. :math:`H_{ij} = 0 \quad (i < j \, \wedge j > r)`
3. :math:`H_{ij} < H_{ii} \quad (i > j, 1 \leq i \leq r)`
4. :math:`r = \mathrm{rank} \mathbf{A}`

If :math:`\mathbf{M}` is full rank, the Hermite normal form :math:`\mathbf{H}` is uniquely determined.


Reference (Japanese)
--------------------

* 伊理 正夫, 線形代数汎論 (朝倉出版, 2009)
