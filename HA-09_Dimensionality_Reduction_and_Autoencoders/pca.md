# Principal Components Analysis (PCA)
<div align="right">noted by Acheng0211(Guojing Huang, SUSTech)</div>

- [Principal Components Analysis (PCA)](#principal-components-analysis-pca)
    - [**1.** Introduction](#1-introduction)
    - [**2.** Finding Principal Components](#2-finding-principal-components)
    - [**3.** Standard PCA](#3-standard-pca)
    - [**4.** Minimizing Reconstruction Error](#4-minimizing-reconstruction-error)
    - [**5.** Autoencoders](#5-autoencoders)
___


### **1.** <big>Introduction</big>
- PCA: most popular instance of second main class of unsupervised learningmethods, projection methods, aka dimensionality-reduction methods
- Aim: fnd a small number of “directions” in input space that explainvariation in input data, re-represent data by projecting along those directions
- Important assumption: variation contains information.
- Data is assumed to be continuous:
    - linear relationship between data and the learned representation
- Handles high-dimensional data
    - If data has thousands of dimensions, can be diffcult for a classifier to deal withOften can be described by much lower dimensional representation
- Useful for:
   - Visualization
   - Preprocessing
   - Modeling - prior for new data
   - Compression


[back to the top](#principal-components-analysis-pca)

### **2.** <big>Finding Principal Components</big>
- Aim to reduce dimensionality
  - linearly project to a much lower dimensional space, $ M \ll D $: $ x \approx U_{pca} z + a $
  where $U_{pca}$ is a $D \times M$ matrix and $z$ is an $M$-dimensional vector.
- Search for orthogonal directions in space with the highest variance
  - project data onto this subspace
- Structure of data vectors is encoded in sample covariance
- subtract the sample mean from each variable: Calculate the **empirical covariance matrix** $$C = \frac{1}{N} \sum^N_{n=1} (x^{(n)} - \bar x)(x^{(n)} - \bar x)^T$$
- Find the $ M $ eigenvectors with largest eigenvalues of $ C $: these are the **principal components**.
- Assemble these eigenvectors into a $ D \times M $ matrix $ U_{pca} $.
- We can now express $ D $-dimensional vectors $ x $ by projecting them to $ M $-dimensional: $z = U_{pca}^T x$

[back to the top](#principal-components-analysis-pca)

### **3.** <big>Standard PCA</big>
- **Algorithm**: to find $ M $ components underlying $ D $-dimensional data

1. Select the top $ M $ eigenvectors of $ C $ (data covariance matrix):
   $$
   C = \frac{1}{N} \sum_{n=1}^{N} (x^{(n)} - \bar{x})(x^{(n)} - \bar{x})^T = U \Sigma U^T \approx U_{1:M} \Sigma_{1:M} U_{1:M}^T
   $$
   where $ U $ is orthogonal, columns are unit-length eigenvectors
   $$
   U^T U = U U^T = I
   $$
   and $ \Sigma $ is a matrix with eigenvalues on the diagonal, representing the variance in the direction of each eigenvector.
   Matrix form: $C = \frac{1}{N}(X-\bar x)(X -\bar x)^T$
2. Project each input vector $ x $ into this subspace, e.g.,

$$
z_j = u_j^T x; \qquad z = U_{1:M}^T x
$$

Let $ U_{pca} = U_{1:M} $. Then we have the principal components

$$
z = U_{pca}^T x
$$

- Two views/derivations:
    - Maximize variance (scatter of green points)
    - inimize error (red-green distance per datapoint)
[back to the top](#principal-components-analysis-pca)

### **4.** <big>Minimizing Reconstruction Error</big>
- PCA: projecting data onto alower-dimensional subspace
- reconstruction between the projection and the original data
$$ J(u, z, b) = \sum_{n} \left\| x^{(n)} - \tilde{x}^{(n)} \right\|^2 $$ where
$$ \tilde{x}^{(n)} = \sum_{j=1}^{M} z_j^{(n)} u_j + \sum_{j=M+1}^{D} b_j u_j $$
- Objective minimized when first $ M $ components are the eigenvectors with the maximal eigenvalues
$$
\begin{align*}
z_{j}^{(n)} &= u_{j}^T x^{(n)}, \quad \forall j = 1, \ldots, M; \\
b_{j} &= \bar{x}^T u_{j}, \quad \forall j = M+1, \ldots, D.
\end{align*}
$$ In the matrix form:
$$
x^{(n)} \approx \tilde{x}^{(n)} = U_{1:M} z^{(n)} + a
$$ where
$$
a = U_{M+1:D} b, \quad b = U_{M+1:D}^T \bar{x}
$$
If the mean $ \bar{x} $ is zero, then $ b = 0 $ and $ a = 0 $.
  
[back to the top](#principal-components-analysis-pca)

### **5.** <big>Autoencoders</big>
- a neural network whose outputs are its own inputs (to minimize reconstruction error)
- **Define**
$$
z = f(Wx); \quad \hat{x} = g(Vz)
$$
- **Goal:**
$$
\min_{W,V} \frac{1}{2N} \sum_{n=1}^{N} \left\| x^{(n)} - \hat{x}^{(n)} \right\|^2
$$
- If $ g $ and $ f $ are linear
$$
\min_{W, V} \frac{1}{2N} \sum_{n=1}^{N} \left\| x^{(n)} - VW x^{(n)} \right\|^2
$$
In other words, the optimal solution is **PCA** for the case when the mean of the data is 0.
- if $g()$ is not linear $\rightarrow$ nonlinear PCA

[back to the top](#principal-components-analysis-pca)

