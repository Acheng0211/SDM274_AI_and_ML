# Linear Regression 
<div style="text-align: right">noted by Acheng0211(Guojing Huang, SUSTech)</div> 

- [What is Regression?](#1-what-is-regression)
- [Linear Regression model](#2-linear-regression-model)
- [Loss Function](#3-loss-function)
- [Optimization](#4-optimization)
    - [Least square solution](#41-least-square-solution)
    - [Gradient decent](#42-gradient-decent)
- [Generalization](#5-generalization)
- [Regularization](#6-regularization)
- [Normalization](#7-normalization)
  - [Min-Max normalization](#71-min-max-normalization)
  - [Mean normalization](#72-mean-normalization)
- [Summary](#summary)
___


### **1.** <big>What is Regression?</big>

Regression is relative to Classification, depending on the predicted variable **_y_pred_**. Normally, we need classification for typed outputs and regression for continuous outputs. However, somehow classification problem can be converted to regression problem.

[back to the top](#linear-regression)

### **2.** <big>Linear Regression model</big>

- True model(unknown) : $t = f(x)$
- Linear Regression model: $y(x) = w_0 + w_1x$

[back to the top](#linear-regression)

### **3.** <big>Loss Function</big>
Standard <font color="red">loss/cost/objective</font> function measures the <font color="red">error</font> between _y_ and the true value _t_.

- Sum of Squares for Error (SSE) $= \displaystyle \sum^{N}_{n=1} (t^{(n)} - y^{(n)})^2$
- Mean Squared Error (MSE) $ = \frac{1}{N} \displaystyle \sum^{N}_{n=1} [t^{(n)} - y^{(n)}]^2$
- Root Mean Squared Error (RMSE) $=\sqrt{\frac{1}{N} \displaystyle \sum^{N}_{n=1} [t^{(n)} - y^{(n)}]^2}$
- Relative Squared Error (RSE) $= \frac{\displaystyle \sum^{N}_{n=1} [t^{(n)} - y^{(n)}]^2}{\displaystyle \sum^{N}_{n=1} [t^{(n)} - \overline{t}]^2}, \overline{t} = \frac{1}{N} \displaystyle \sum^{N}_{n=1} t^{(n)} $
- Mean Absolute Error (MAE) $= \frac{1}{N} \displaystyle \sum^{N}_{n=1} |{t^{(n)} - y^{(n)}}|$
- Relative Absolute Error (RAE) $= \frac{\displaystyle \sum^{N}_{n=1} |t^{(n)} - y^{(n)}|}{\displaystyle \sum^{N}_{n=1} |t^{(n)} - \overline{t}|}, \overline{t} = \frac{1}{N} \displaystyle \sum^{N}_{n=1} t^{(n)} $

[back to the top](#linear-regression)

### **4.** <big>Optimization</big>
- Preprocess: incorporate the bias $w_0$ into **w** by using $x_0=1$ (Add an **1** to input **x**) . Then, **x** = $\left[\begin{matrix}1 \\ x \end{matrix} \right]$
- Linear regression model: $y(x) = w^T\mathbf{x}$
- MSE loss: $l(w) = \frac{1}{2N} \displaystyle \sum^{N}_{n=1} [t^{(n)} - y({x^{(n)}})]^2$, convex

    ####  4.1 Least square solution
    1. let the gredient equal to 0, to find the minima: $\nabla l(w) = -\frac{1}{N} \displaystyle \sum^{N}_{n=1} (t^{(n)} - w^T\mathbf{x}{(n)})\mathbf{x}^{(n)}$ = 0
    2. then we get: $w = (\mathbf{x}^{T}\mathbf{x})^{-1}\mathbf{x}^{T}t$

    #### 4.2 Gradient decent
    - let gradient decrease to the smallest through iteration: Initalize at one point, calculate its gradient and move in the opposite direction. 

    - Protocol:
        1. initialize $w$ (randomly)
        2. repeatedly update $w$ based on the gradient, $\lambda$ is the <font color="red">learning rate</font>

    - 3 ways of gradient descent:
        1. Stochastic Gradient Descent (SGD): 
            - Randomly shuffle and pick one sample $(x^{(n)}, t^{(n)})$ in the training set, then update by $w \leftarrow w + \lambda[t^{(n)} - y({x^{(n)}})]\mathbf{x}^{(n)}$.
            - Update the parameters for **each sample** in turn, update breaks as error approaches zero.
        2. Batch Gradient Descent (BGD):
            - update by $w \leftarrow w + \lambda \frac{1}{N}\displaystyle \sum^{N}_{n=1}[t^{(n)} - y({x^{(n)}})]\mathbf{x}^{(n)}$ (equivalent to matrix form: $w \leftarrow w + \lambda \frac{1}{N}\mathbf{X}^{T}(t - \mathbf{X}w)$).
            - **Average** update across **every sample**, update breaks as error approaches zero.
        3. Mini-Batch Gradient Descent
            - Shuffle the training set and partition into a number of mini-batches($m$)
            - update by $w \leftarrow w + \lambda \frac{1}{m}\displaystyle \sum_{n \in B_j}[t^{(n)} - y({x^{(n)}})]\mathbf{x}^{(n)}$ (equivalent to matrix form: $w \leftarrow w + \lambda \frac{1}{m}\mathbf{X}^{T}_{B_i}(t_{B_i} - \mathbf{X}_{B_i}w)$).

- Gradient decent vs least square approach

|Gradient decent|least square approach|
|----|----|
|Need to select a learning rate $\lambda$|Does not require selecting a learning rate|
|multiple step iterations|One-time computation, but requies compueting $(\mathbf{X}^T\mathbf{X})^{-1}$|
|Performs well even with a large number of $d$ features|large computation complexity $O(d^3)$, acceptable when $d < 10000$|
|Suitable for various types of models|Only suitable for linear models|

[back to the top](#linear-regression)

### **5.** <big>Generalization</big>
-  **Generalization** = model's ability to predict the held out data

[back to the top](#linear-regression)

### **6.** <big>Regularization</big>
- One way of dealing with overifitting is to encourage the weights to be small
- Standard regularization approach:
    $l(w) = \displaystyle \sum^{N}_{n=1}[t^{(n)} - y({x^{(n)},w})]^2 + \alpha w^{T}w$, $\alpha$ is a hyper-parameter
- also called **ridge regression**

[back to the top](#linear-regression)

### **7.** <big>Normalization</big>
- Improving model accuracy: Comparability in values between features across different dimensions can significantly enhance the accuracy of model learning.
- Accelerating learning convergence: Searching for the optimum becomes notably smoother, making it easier for the model to converge correctly to the optimal solution.

    #### 7.1 Min-Max normalization
    $$x^{*} = \frac{x - x_{min}}{x_{max} - x_{min}}$$
    - Maps the data along any dimension to [0,1]
    - The purpose of **min-max normalization**: make the impact of each feature compatible, which involves scaling transformations of the features
    - Normalizing data will alter the distribution of the feature data.
    #### 7.2 Mean normalization
    $$x^{*} = \frac{x - \mu}{\sigma}$$
    where
    $$\mu = \frac{1}{N} \displaystyle \sum^{N}_{n=1} x^{(i)},  \sigma^2 = \frac{1}{N} \displaystyle \sum^{N}_{n=1} (x^{(i)} - \mu)^2$$
    - The data becomes **zero mean** and **unit variance**
    - **Mean normalization**: aims to make different features comparable to each other
    - The distribution of the feature data remains unchanged

[back to the top](#linear-regression)

###  <big>Summary</big>
- Data fits: linear regressio model may not be the best selected model, sometimes underfit or overfit.
- One method of assessing fit: test **generalization** = model's ability to predict the held out data
- **Optimization** is essential: stochastic and batch iterative approaches; analystic when available 

[back to the top](#linear-regression)


