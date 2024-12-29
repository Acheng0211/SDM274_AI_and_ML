# Support Vector Machine
<div align="right">noted by Acheng0211(Guojing Huang, SUSTech)</div>

- [Support Vector Machine](#support-vector-machine)
    - [**1.** Introduction](#1-introduction)
    - [**2.** Max-margin Classification](#2-max-margin-classification)
    - [**3.** Linear SVM](#3-linear-svm)
    - [**4.** Nonlinear-SVM](#4-nonlinear-svm)
    - [**5.** Algorithm](#5-algorithm)
    - [**6.** Nonlinear Decision Boundaries](#6-nonlinear-decision-boundaries)
    - [**7.** Kernels](#7-kernels)
    - [**8.** Summary](#8-summary)
___


### **1.** <big>Introduction</big>
- Supervised learning
- binary classification: Tiny change frombefore: instead of using $t = l$ and $t = 0$ for positive and negative class, we will use $t = l$ for the positive and $t = -l$ for the negative class

[back to the top](#support-vector-machine)

### **2.** <big>Max-margin Classification</big>
- focus on the boundary point instead of all the points
- learn a boundary that leads to the largest margin (buffer) from points on both sides
- support vectors: subset of vectors that support (determine boundary)

[back to the top](#support-vector-machine)

### **3.** <big>Linear SVM</big>
- Binary and Linear Separable Classification, Linear Classifier with Maximal Margin
- Max margin classifier: Inputs in margin are of unknown class
\[
y = 
\begin{cases} 
1 & \text{if } w^T x + b \geq 1 \\ 
-1 & \text{if } w^T x + b \leq -1 \\ 
\text{Undefined} & \text{if } -1 \leq w^T x + b \leq 1 
\end{cases}
\]  Equivalently, can write above condition as:
\[
(w^T x + b) y \geq 1
\]
- The vector \( w \) is orthogonal to the +1 plane.
If \( u \) and \( v \) are two points on that plane, then \( w^T (u - v) = 0 \)
- Same is true for -1 plane
- Also: for point \( x_+ \) on +1 plane and \( x_- \) nearest point on -1 plane:
\[
x_+ = \lambda w + x_-, \qquad \lambda = \frac{2}{w^T w}
\]
- Define the margin \( M \) to be the distance between the +1 and -1 planes
- We can now express this in terms of \( w \) to maximize the margin
  - Or, equivalently, we minimize the length of \( w \)
\[
M = \|x_+ - x_-\| = \|\lambda w\| = \lambda \sqrt{w^T w} = 2 \frac{\sqrt{w^T w}}{w^T w} = \frac{2}{\|w\|}
\]
- We can search for the optimal \( w \) and \( b \) by finding a solution that:
  1. Correctly classifies the training examples: \( \{(x^{(i)}, t^{(i)})\}_{i=1}^N \)
  2. Maximize the margin (same as minimizing \( w^T w \))
   primal formulation:
    \[
    \min_{w,b} \frac{1}{2} \|w\|^2
    \] s.t. \( (w^T x^{(i)} + b) t^{(i)} \geq 1, \quad \forall i = 1, \ldots, N \)
- Apply <big>Lagrange multipliers</big>: formulate equivalent problem
  - constrained optimization with **equality constraints**: $$\min_x f(x), \ \text{s.t.} \ h_i(x) = 0, \forall i=1,\ldots,m $$
    - **Lagrangian functon**: $\displaystyle l(x,\mu) = f(x) + \sum^m_{i=1} \mu_i h_i(x)$, $\mu_i$ are the **Langrange multipliers**
    - add a penalty term: $\displaystyle \min_x f(x) + \sum_{i=1}^m p_i(x), \ p_i(x) = \max_{\mu_i \neq 0} \mu_i h_i(x)$
    - Then, $\displaystyle \min_x \{f(x) + \sum_{i=1}^m p_i(x)\} = \min_x \max_{\mu_i \neq 0} l(x,\mu)$
  - constrained optimization with **inequality constraints**: $$\min_x f(x), \ \text{s.t.} \ g_i(x) \leq 0, \forall i=1,\ldots,n $$
    - **Lagrangian functon**: $\displaystyle l(x,\alpha) = f(x) + \sum^m_{i=1} \alpha_i h_i(x)$, $\alpha_i$ are the **Langrange multipliers**
    - add a penalty term: $\displaystyle \min_x f(x) + \sum_{i=1}^m p_i(x), \ p_i(x) = \max_{\alpha_i \geq 0} \alpha_i g_i(x)$
    - Then, $\displaystyle \min_x \{f(x) + \sum_{i=1}^m p_i(x)\} = \min_x \max_{\alpha_i \geq 0} l(x,\alpha)$
    - $\text{Primal formulation:} \ \min_x \max_{\alpha_i \geq 0} l(x,\alpha), \quad \text{Dual formulation:} \ \max_{\alpha_i \geq 0} \min_x l(x,\alpha)$ 
    $\text{KKT Condition:} \displaystyle \max_{\alpha_i \geq 0} \min_x l(x,\alpha) \leq \min_x \max_{\alpha_i \geq 0} l(x,\alpha)$

- Training SVM by Maximizing
\[
L = \max_{\alpha_i \geq 0} \left\{ \sum_{i=1}^N \alpha_i - \frac{1}{2} \sum_{i,j=1}^N t^{(i)} t^{(j)} \alpha_i \alpha_j (x^{(i)})^T x^{(j)} \right\} \qquad \text{subject to:} \ \alpha_i \geq 0, \ \sum_{i=1}^N \alpha_i t^{(i)} = 0
\]
  - The Weights are $\displaystyle w = \sum_{i=1}^N \alpha_i t^{(i)} x^{(i)}$
  - Only a small subset of \( \alpha_i \)'s will be nonzero (why?), and the corresponding \( x^{(i)} \)'s are the support vectors \( S \).
  - The prediction \( y \) for a new example \( x \) is given by:
    \[
    y = \operatorname{sign} \left[ b + x \cdot \left( \sum_{i=1}^N \alpha_i t^{(i)} x^{(i)} \right) \right] = \operatorname{sign} \left[ b + x \cdot \left( \sum_{i \in S} \alpha_i t^{(i)} x^{(i)} \right) \right]
    \]
- Optimal Value for $b$
  - penalty-term: $\displaystyle \max_{\alpha_i \geq 0} \ \alpha_i[1-(w^Tx^{i}+b)]t^{(i)}= \begin{cases} 0 & \text{if } (w^T x^{(i)} + b)t^{(i)} \geq 1 \\ \infty & \text{otherwise} \end{cases}$
  - optimal b = $b^* = t^{(i)} - w^{*T}x^{(i)} = t^{(i)} - \sum^N_{j=1} \alpha^*_j t^{(j)}(x^{(j)} \cdot x^{(i)})$  
  - Averaging over all support vectors:
  \[
  b^* = \frac{1}{|S|} \sum_{i \in S} \left[ t^{(i)} - w^{*T} x^{(i)} \right] = \frac{1}{|S|} \sum_{i \in S} \left[ t^{(i)} - \sum_{j=1}^N \alpha_j^* t^{(j)} \left( x^{(j)} \cdot x^{(i)} \right) \right]
  \] or equivalently,
  \[
  b^* = \frac{1}{\sum_{i=1}^N \delta(\alpha_i > 0)} \sum_{i=1}^N \left[ t^{(i)} - \sum_{j=1}^N \alpha_j^* t^{(j)} \left( x^{(j)} \cdot x^{(i)} \right) \right]
  \] Where \( \delta(\alpha_i > 0) \) is an indicator function, taking value 1 if \( \alpha_i > 0 \) and 0 otherwise.

[back to the top](#support-vector-machine)

### **4.** <big>Nonlinear-SVM</big>
- Introduce Slack Variables \( \xi_i \):
\[
\min_{w,b,\xi} \frac{1}{2} \|w\|^2 + \lambda \sum_{i=1}^{N} \xi_i
\]subject to: \( \xi_i \geq 0 \); \( \  t^{(i)}(w^T x^{(i)} + b) \geq 1 - \xi_i \), \(\forall i = 1, \ldots, N \)
- Lagrangian Function
\[
J(w, b, \xi; \alpha, \mu) = \frac{1}{2} \|w\|^2 + \lambda \sum_{i=1}^{N} \xi_i + \sum_{i=1}^{N} \alpha_i \left[ 1 - \xi_i - t^{(i)}(w^T x^{(i)} + b) \right] + \sum_{i=1}^{N} \mu_i (-\xi_i)
\] \[
\max_{\alpha_i \geq 0, \mu_i \geq 0} \min_{w,b,\xi} J(w, b, \xi; \alpha, \mu)
\]
- Conditions for \( \alpha_i \)
  1. \( \alpha_i = 0 \) if \( t^{(i)}(w^T x^{(i)} + b) \geq 1 \) (sample \( i \) is on the correct side with \( \xi_i = 0 \))
  2. \( 0 < \alpha_i < \lambda \) if \( t^{(i)}(w^T x^{(i)} + b) = 1 \) (sample \( i \) is a support vector)
  3. \( \alpha_i = \lambda \) if \( t^{(i)}(w^T x^{(i)} + b) \leq 1 \) (sample \( i \) is on the wrong side with \( \xi_i \neq 0 \))

[back to the top](#support-vector-machine)

### **5.** <big>Algorithm</big>
- SMO (Sequential Minimal Optimization) Algorithm  
  - Select two variables $\alpha_i$ and $\alpha_j$ at a time by treating others asconstants, and solve the corresponding quadratic programming problem 
    - Actually, since we have an equality constraint, we only need to solve a quadraticprogramming problem with only one variable $\alpha_i$
- Convex Optimization Algorithm
  - Equivalent Quadratic Programming in the Matrix Form:
\[
\min_\alpha \frac{1}{2} \alpha^T Q \alpha + p^T \alpha
\] s.t. \( t^T \alpha = 0 \); \( 0 \leq \alpha_i \leq \lambda \), for \( i = 1, \cdots, N \)
  - For this formula,
\[
Q(i, j) = t^{(i)} t^{(j)} (x^{(i)} \cdot x^{(j)}), \quad p(i) = -1
\]
  - So,
\[
Q = \left( \begin{bmatrix} t^{(1)} \\ \vdots \\ t^{(N)} \end{bmatrix} \begin{bmatrix} t^{(1)} & \cdots & t^{(N)} \end{bmatrix} \right) \odot \left( \begin{bmatrix} x^{(1)} \\ \vdots \\ x^{(N)} \end{bmatrix} \begin{bmatrix} x^{(1)} & \cdots & x^{(N)} \end{bmatrix} \right) = (t t^T) \odot (X X^T)
\]

> \( \odot \) denotes element-wise product, also known as **Hadamard product**.

[back to the top](#support-vector-machine)


### **6.** <big>Nonlinear Decision Boundaries</big>
- Steps:
  1. Map data into feature space: 
     \[
     x \rightarrow \phi(x)
     \]
  2. Replace dot products between inputs with feature points:
     \[
     x^{(i)} \cdot x^{(j)} \rightarrow \phi(x^{(i)}) \cdot \phi(x^{(j)})
     \]
  3. Find linear decision boundary in feature space
- Mapping to a feature space can produce problems:
  - High computational burden due to high dimensionality
  - Many more parameters

[back to the top](#support-vector-machine)


### **7.** <big>Kernels</big>
- Dual formulation only assigns parameters to samples, not features
- Kernel Trick: Dot-products in feature space can be computed as a 
  \[
  K(x^{(i)}, x^{(j)}) = \phi(x^{(i)}) \cdot \phi(x^{(j)})
  \]
- Idea: Work directly on \( x \), avoid having to compute \( \phi(x) \). 
- Calculation for particular mapping \( \phi(x) \) implicitly maps to high-dimensional space.
- Examples of Kernels: Kernels measure similarity.
    1. **Polynomial**:
    \[
    K(x^{(i)}, x^{(j)}) = (x^{(i)^T} x^{(j)} + 1)^d
    \]
    > where \( d \) is the degree of the polynomial, e.g., \( d = 2 \) for quadratic.
    2. **Gaussian**:
    \[
    K(x^{(i)}, x^{(j)}) = \exp\left(-\frac{\|x^{(i)} - x^{(j)}\|^2}{2\sigma^2}\right)
    \]
    3. **Sigmoid**:
    \[
    K(x^{(i)}, x^{(j)}) = \tanh(\beta (x^{(i)^T} x^{(j)}) + a)
    \]
- Why is this useful?
  - Rewrite training examples using more complex features.
  - Dataset not linearly separable in original space may be linearly separable in higher dimensional space.
- Mercer's Theorem (1909): Any reasonable kernel corresponds to some feature space.
- Reasonable means that the Gram matrix is positive definite:
  \[
  K_{ij} = K(x^{(i)}, x^{(j)})
  \]
- Feature space can be very large:
  - Polynomial kernel \( (1 + (x^{(i)})^T x^{(j)})^d \) corresponds to feature space exponential in \( d \).
  - Gaussian kernel has infinitely dimensional features.
- Linear separators in these super high-dimensional spaces correspond to highly nonlinear decision boundaries in input space.
- $x^{(j)} \cdot x^{(i)} \rightarrow K(x^{(j)},x^{(i)})$

[back to the top](#support-vector-machine)


### **8.** <big>Summary</big>
- Advantages:
  - Kernels allow very flexible hypotheses
  - Poly-time exact optimization methods rather than approximate methods
  - Soft-margin extensions permits mis-classified examples
  - Excellent results (1.1% error rate on handwritten digits vs. LeNet's 0.9%)
- Disadvantages: must choose kernel parameters
- Difference between logistic regression and SVMs
- Maximum margin principle
- Target function for SVMs
- Slack variables for mis-classified points
- Kernel trick allows non-linear generalizations
  
[back to the top](#support-vector-machine)
