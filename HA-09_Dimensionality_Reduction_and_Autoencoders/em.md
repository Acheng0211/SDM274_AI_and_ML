# Ensemble Methods
<div align="right">noted by Acheng0211(Guojing Huang, SUSTech)</div>

- [Ensemble Methods](#ensemble-methods)
    - [**1.** Introduction](#1-introduction)
    - [**2.** Bootstrap Estimation](#2-bootstrap-estimation)
    - [**3.** Bagging](#3-bagging)
    - [**4.** Boosting (AdaBoost)](#4-boosting-adaboost)
    - [**5.** Random/Decision Forests](#5-randomdecision-forests)
    - [**6.** Mixture of Experts](#6-mixture-of-experts)
___


### **1.** <big>Introduction</big>
- Ensemble of classifers is a set of classifers whose individual decisions arecombined in some way to classify new examples
- Simplest approach:
    1. Generate multiple classifiers
    2. Each votes on test instance
    3. Take majority as classification
- Classifers are different due to different sampling of training data, orrandomized parameters within the classification algorithm
- Aim: Take simple mediocre algorithm and transform it into a superclassifier without requiring any fancy new algorithm
- Differ in training strategy, and combination method
    - **Parallel training** with different training sets
    **Bagging** (bootstrap aggregation)-train separate models on overlappingtraining sets, average their predictions
    - **Sequential training**, iteratively re-weighting training examples so currentclassifier focuses on hard examples: boosting
    - Parallel training with objective encouraging division of labor: **mixture of experts**
- Notes:
    - Also known as meta-learning
    - **Typically applied to weak models**, such as decision stumps (single-nodedecision trees), or linear classifers
- Minimize two sets of errors:
    1. Variance: Error from sensitivity to small fuctuations in the training set
    Variance reduction: If the training sets are completely independent, itwill always help to average an ensemble because this will reducevariance without affecting bias (e.g., bagging)
    > reduce sensitivity to individual data points
    2. Bias: Erroneous assumptions in the model
    Bias reduction: For simple models, average of models has much greatercapacity than single model(e.g., hyperplane classifers, Gaussian.densities)
    > Averaging models can reduce bias substantially by increasing capacity,and control variance by fitting one component at a time (e.g., boosting)
- Justification
  - Independent errors: Prob $k$ of $N$ classifiers (independent error rate $\epsilon$) wrong: $$P(num \ errors = k) = \begin{pmatrix} N \\ k \end{pmatrix} \epsilon^k(1-\epsilon)^{N-k}$$
  - N is bigger $\rightarrow$ $P$ is smaller

[back to the top](#ensemble-methods)

### **2.** <big>Bootstrap Estimation</big>
- Repeatedly draw $n$ samples from $N$
- For each set of samples, estimate a statistic
- The bootstrap estimate is the mean of the individual estimates
- Used to estimate a statistic (parameter) and its variance
> **Bagging**: **b**ootstrap **agg**regation (Breiman 1994)

[back to the top](#ensemble-methods)

### **3.** <big>Bagging</big>
- get $M$ bootstrap samples and average them: $\displaystyle y^M_{bag}(x)=\frac{1}{M}\sum^M_{m=1}y_m(x)$
  - regression $\rightarrow$ average predictions
  - classfication $\rightarrow$ average class probabilities or take the majority vote if only hard outputs available
- approximates the Bayesian posterior mean
- Each bootstrap sample is drawn with replacement, so each one containssome **duplicates of certain training points** and leaves out other trainingpoints completely

[back to the top](#ensemble-methods)

### **4.** <big>Boosting (AdaBoost)</big>
- Also works set by manipulating training set, but classifers trained sequentially
- Each classifier trained given knowledge of the performance of previouslytrained classifers: focus on hard examples
- Final classifer: weighted sum of component classifiers
- Procedure:
  - First train the base classifer on all the training data with equal importanceweights on each case.
  - Then re-weight the training data to emphasize the hard cases and train asecond model.
  - Keep training new models on re-weighted data
  - Finally, use a weighted committee of all the models for the test data.
- Weight on Example \( n \) for Classifier \( m \): \( w_n^m \)
- Cost Function for Classifier \( m \):
\[
J_m = \sum_{n=1}^N w_n^m [y_m(x^n) \neq t^{(n)}] = \sum \text{weighted errors}
\]
> 1 if error, 0 otherwise.
- Weighted Error Rate of a Classifier
\[
\epsilon_m = \frac{J_m}{\sum_n w_n^m}
\]
- The Quality of the Classifier is
\[
\alpha_m = \ln\left(\frac{1 - \epsilon_m}{\epsilon_m}\right)
\]
> It is zero if the classifier has a weighted error rate of 0.5 and infinity if the classifier is perfect.
- The Weights for the Next Round are Then
\[
w_n^{m+1} = \exp\left(-\frac{1}{2} t^{(n)} \sum_{i=1}^m \alpha_i y_i(x^{(n)})\right) = w_n^m \exp\left(-\frac{1}{2} t^{(n)} \alpha_m y_m(x^{(n)})\right)
\]
  - If the prediction of example \( n \) is correct, then \( t^{(n)} y_m^{(n)} > 0 \). Then the weight for this example decays.
  - On the other hand, if the prediction of example \( n \) is wrong, then \( t^{(n)} y_m^{(n)} < 0 \). Then the weight for this example grows.    
> If at this round, the weighted error rate is lower than 0.5, then the quality \( \alpha_i \) of the classifier will be positive.

  
- AdaBoost Algorithm
  - **Input**: \(\{x^{(n)}, t^{(n)}\}_{n=1}^{N}\), and a learning procedure **WeakLearn** that produces a classifier \(y(x)\).
  - **Initialize example weights**: \(w_{n}^{1}(x) = \frac{1}{N}\).
  - Iterative Process for \(m = 1\) to \(M\)
  1. **Train Classifier**: 
     \[
     y_{m}(x) = \text{WeakLearn}(x, t, w), \text{ fit classifier by minimizing}
     \]
     \[
     J_{m} = \sum_{n=1}^{N} w_{n}^{m} [y_{m}(x^{n}) \neq t^{(n)}]
     \]
  2. **Compute Unnormalized Error Rate**:
     \[
     \epsilon_m = \frac{J_m}{\sum_n w_n^m}
     \]
  3. **Compute Classifier Quality**:
     \[
     \alpha_m = \ln\left(\frac{1 - \epsilon_m}{\epsilon_m}\right)
     \]
  4. **Update Data Weights**:
     \[
     w_{n}^{m+1} = w_{n}^{m} \exp\left(-\frac{1}{2} t^{(n)} \alpha_m y_{m}(x^{(n)})\right)
     \]
  5. **Final Model**:
    \[
    y(x) = \text{sign}\left(\sum_{m=1}^{M} \alpha_m y_{m}(x)\right)
    \]  
- Popular choices of base classifier for boosting and other ensemble methods.
    - Linear classifers
    - Decision trees

[back to the top](#ensemble-methods)

### **5.** <big>Random/Decision Forests</big>
- Definition: Ensemble of decision trees
- Algorithm:
    - Divide training examples into multiple training sets (bagging)Train a decision tree on each set (can randomly select subset of variables to。consider)
    - Aggregate the predictions of each tree to make classification decision (e.g., can0choose mode vote)

[back to the top](#ensemble-methods)

### **6.** <big>Mixture of Experts</big>
![MoE](/HA-09_Dimensionality_Reduction_and_Autoencoders/figures/MoE.png "MoE")
- Gating Network **encourages specialization**(local experts) rather than cooperation.
  \[
  y(x) = \sum_m g_m(x) y_m(x)
  \]
  > where \( g_m \) refers to the gating network.
- Cost function designed to make each expert estimate desired output **independently**
- **Gating network softmax over experts**: stochastic selection of who is thetrue expert for given input
- Allow each expert to produce **distribution over outputs**

[back to the top](#ensemble-methods)
