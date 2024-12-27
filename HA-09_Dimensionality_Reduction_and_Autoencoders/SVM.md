# Support Vector Machine
<div align="right">noted by Acheng0211(Guojing Huang, SUSTech)</div>

- [Support Vector Machine](#support-vector-machine)
    - [**1.** Introduction](#1-introduction)
    - [**2.** Max-margin Classification](#2-max-margin-classification)
    - [**3.** Linear SVM](#3-linear-svm)
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
\[
\min_{w,b} \frac{1}{2} \|w\|^2
\] s.t. \( (w^T x^{(i)} + b) t^{(i)} \geq 1, \quad \forall i = 1, \ldots, N \)
- This is called the primal formulation of Support Vector Machine (SVM)
- Apply Lagrange multipliers: formulate equivalent problem

[back to the top](#support-vector-machine)
