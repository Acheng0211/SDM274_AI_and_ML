# Classification and Logistic Regression
<div align="right">noted by Acheng0211(Guojing Huang, SUSTech)</div>

- [Classification and Logistic Regression](#classification-and-logistic-regression)
    - [**1.** Classification problem](#1-classification-problem)
    - [**2.** Perceptron](#2-perceptron)
    - [**3.** Metrics](#3-metrics)
    - [**4.** Logistic regression](#4-logistic-regression)
    - [**5.** Summary](#5-summary)
___


### **1.** <big>Classification problem</big>

- Features in common: categorical **outputs**, called **labels** 
- Classification: Assigning each input vector to one of a finite number of labels.(Binary/Multiclass ~ = two/multiple possible labels) 


[back to the top](#classification-and-logistic-regression)

### **2.** <big>Perceptron</big>

- perceptron(linear classifier): $y = sign(z(x,w)) = sign(w_0 + w_1x) = sign(w^T\bar{x}) \in{[-1,1]}$
![perceptron](./figures/perceptron.png "perceptron")
- **decision boundary**(hyperplane): $w_0 + w_1x_1 + ... + w_nx_n = 0$. 1D: threshold, 2D: line, 3D: plane
- Loss function: $$ l(w) = \left\{ \begin{matrix} 0 \qquad if \quad y(x^{(n),w})=t^{(n)}\\ positive \quad if \quad y(x^{(n),w})\neq t^{(n)} \end{matrix} \right. = [-t^{(n)}z(x^{(n)},w)]_{+}$$
    $$\nabla l(w) = \left\{ \begin{matrix} 0 \qquad if \quad y(x^{(n),w})=t^{(n)}\\ -t^{(n)} \bar{x}^{(n)} \quad if \quad y(x^{(n),w})\neq t^{(n)} \end{matrix} \right.$$
- Perceptron can only deal with linearly separable problem
- Causes of non perfect separation:
  - Model is too simple (perceptron)
  - Noises in the inputs (i.e., data attributes)
  - Simple features that do not account for all variations
  - Errors in data targets (mis-labellings)

[back to the top](#classification-and-logistic-regression)

### **3.** <big>Metrics</big>
![metrics](./figures/metrics.png "metrics")

- **Accuracy**: gives the percentage of correct classifications $$A = \frac{TP+TN}{TP+FP+FN+TN}$$
- **Recall**: is the fraction of relevant instances that are retrieved $$R = \frac{TP}{TP+FN} = \frac{TP}{all\ groundtruth\ instances}$$
- **Precision**: is the fraction of retrieved instances that are relevant $$P = \frac{TP}{TP+FP} = \frac{TP}{all\ predicted\ true}$$
- **F1 score**: harmonic mean of precision and recall $$F1 = 2\frac{P·R}{P+R}$$

[back to the top](#classification-and-logistic-regression)

### **4.** <big>Logistic regression</big>
- $y = \sigma(z) = \frac{1}{1+exp(-z)} =  \sigma(w^T\bar{x}) \in {[0,1]}$ ($sign(·) \rightarrow$  sigmoid (or logistic) functions)
- 1D: $y(x) = \sigma(w_0 + w_1x)$, $w_0$ indicates the $x(= -w_0)$ coorinates while $y = 0.5$; $w_1$ indicates the curvature (bigger $\rightarrow$ more tilted)
- Decision boundary: $$p(C=1|x,w) = p(C=0|x,w) = \sigma(w^T\bar{x}) = 0.5, w^T\bar{x}=0$$
- Detour
  - $$Probability:p(event|distribution),\ Likelyhood: L(distribution|data)$$
  - Maximum likelyhood extimation
    $$w^*=\arg\max_wL(x,w)=\prod_{i=1}^Np(x^{(i)},w)$$
  - maximum log-likelyhood
    $$w^*=\arg\max_wl(x,w)=\sum_{i=1}^N log(p(x^{(i)},w))$$
- likelyhood: 
  $$L(w) = \prod_{i=1}^Np(t^{(i)}|x^{(i)};w), \ \begin{aligned}
p(t^{(i)}|x^{(i)};w) & =\quad p(C=1|x^{(i)};w)^{t^{(i)}}p(C=0|x^{(i)};w)^{1-t^{(i)}} \\
 & =\quad\left(p(C=1|x^{(i)};w)\right)^{t^{(i)}}\left(1-p(C=1|x^{(i)};w)\right)^{1-t^{(i)}}
\end{aligned} $$
    loss function:
    $$\ell_{\text{log}}(w) = -\log L(w)= -\sum_{i=1}^{N} t^{(i)} \log y(x^{(i)}, w) - \sum_{i=1}^{N} (1 - t^{(i)}) \log \left(1 - y(x^{(i)}, w)\right)$$
    gradient: 
    $$\nabla l(w) = - \sum_{i=1}^{N}[t^{(i)}-y(x^{(i)},w)]\bar{x}^{(i)}$$
- Regulation: helps avoid large weights and overfitting
$$ min_{w} \quad \tilde{l}(w) = -\log \left( p(w) \prod_{i} p(t^{(i)} | x^{(i)}; w) \right) 
$$
    - define prior: normal distribution, zero mean and identity covariance
$$ w \sim N(0, \alpha^{-1} I)
$$
$$ p(w) = \prod_{j=1}^{d+1} \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{w_j^2}{2\sigma^2}\right) \quad \text{where} \quad \sigma^2 = \alpha^{-1}
$$
  - This prior pushes parameters towards zero. Plug above and obtain
$$ \tilde{l}(w) = l(w) + \sum_{j=1}^{d+1} \left( \frac{1}{2} \log(2\pi\sigma^2) + \frac{w_j^2}{2\sigma^2} \right)
$$
  - Including this prior the new gradient is
$$ \nabla \tilde{l}(w) = \nabla l(w) + \alpha w
$$
- Tuning hyper-parameters:
   - **Never Use Test Data For Tuning The Hyper-Parameters** 
   - We can divide the set of training examples into two disjoint sets: training and validation
   - Use the first set (i.e., training) to estimate the weights $w$ for different values of $\alpha$ 
   - Use the second set (i.e., validation) to estimate the best $\alpha$, by evaluating howwell the classifer does on this second set
   - This tests how well it generalizes to unseen data
- Cross-validation
  - Leave-$p$-out cross-validation:
    - We use $p$ observations as the validation set and the remaining observations as thetraining set.
    - This is repeated on all ways to cut the original training set.
    - It requires $C^p_N$ for a set of $N$ examples
  - Leave-$1$-out cross-validation: When $p=1$, does not have this problem
  - *k*-fold cross-validation:
    - The training set is randomly partitioned into $k$ equal size subsamples.
    - Of the $k$ subsamples, a single subsample is retained as the validation data fortesting the model, and the remaining $k-1$ subsamples are used as training data.
    - The cross-validation process is then repeated $k$ times (the folds).
    - The $k$ results from the folds can then be averaged (or otherwise combined) toproduce a single estimate


[back to the top](#classification-and-logistic-regression)

### **5.** <big>Summary</big>
- Advantages:
    - Easily extended to multiple classeso Natural probabilistic view of class predictions
    - Quick to train
    - Fast at classifcation
    - Good accuracy for many simple data sets
    - Resistant to overftting
    - Can interpret model coefficients as indicators of feature importance
- Less good:
    - Linear decision boundary (too simple for more complex problems)

[back to the top](#classification-and-logistic-regression)