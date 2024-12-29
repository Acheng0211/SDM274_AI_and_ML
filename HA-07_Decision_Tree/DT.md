# Decision Tree
<div align="right">noted by Acheng0211(Guojing Huang, SUSTech)</div>

- [Decision Tree](#decision-tree)
    - [**1.** Introduction](#1-introduction)
    - [**2.** Classification \& Regression](#2-classification--regression)
    - [**3.** Learning Decision Trees](#3-learning-decision-trees)
    - [**4.** Decision Tree Construction Algorithm](#4-decision-tree-construction-algorithm)
    - [**5.** Selecting the right threshold](#5-selecting-the-right-threshold)
    - [**6.** Summary](#6-summary)
___


### **1.** <big>Introduction</big>
- Internal nodes test attributes
- Branching is determined by attribute value
- Leaf nodes are outputs (class assignments)
- Procedure；
  - Choose an attribute on which to descend ateach level
  - Condition on earlier (higher) choices
  - Generally, restrict only one dimension at a time
  - Declare an output value when you get to the bottom

[back to the top](#decision-tree)

### **2.** <big>Classification & Regression</big>
- Each path from root to a leaf defines a region \( R_m \) of input space. Let \( \{(x^{(m_1)}, t^{(m_1)}), \ldots, (x^{(m_k)}, t^{(m_k)})\} \) be the training examples that fall into \( R_m \).
- Classification Tree:
  - Discrete Output
  - Leaf value \( y^m \) typically set to the most common value in \( [t^{(m_1)}, \ldots, t^{(m_k)}] \)
- Regression Tree:
  - Continuous Output
  - Leaf value \( y^m \) typically set to the mean value in \( [t^{(m_1)}, \ldots, t^{(m_k)}] \)
- different type of input-and-output case:
  - Discrete: DT can express any function of the input attributes
  - Continuous: can approximate any function arbitrarily closely

[back to the top](#decision-tree)

### **3.** <big>Learning Decision Trees</big>
- Resort to a greedy heuristic: Start from an empty decision tree $\rightarrow$ Split on next best attribute $\rightarrow$ Recurse
- Choosing a good attribute
  - Deterministic: good (all are true or false, just one class in the leaf)
  - Uniform distribution: bad (all classes in leaf equally probable)
- Quantifying Uncertainty
  - Entropy $H$: $\displaystyle H(X)=-\sum_{x \in X} p(x)log_2p(x)$
    - "High Entropy
      - Variable has a uniform like distribution
      - Flat histogram
      - Values sampled from it are less predictable
    - "Low Entropy"
      - Distribution of variable has many peaks and valleys
      - Histogram has many lows and highs
      - Values sampled from it are more predictable
  - Conditonal Entropy: $\displaystyle H(Y \mid X) = \sum_{x \in X} p(x) H(Y \mid X = x) = - \sum_{x \in X} \sum_{y \in Y} p(x, y) \log_2 p(y \mid x)$
    - H is always non-negative
    - **Chain rule**: $ H(X, Y) = H(X|Y) + H(Y) = H(Y|X) + H(X) $
    - If \( X \) and \( Y \) are independent, then \( X \) doesn't tell us anything about \( Y \): $ H(Y|X) = H(Y) $
    - But \( Y \) tells us everything about \( Y \): $ H(Y|Y) = 0 $
    - By knowing \( X \), we can only decrease uncertainty about \( Y \): $ H(Y|X) \leq H(Y) $
  - Information gain (in $Y$ due to $X$): $IG(Y|X) = H(Y) - H(Y|X)$
    - If $X$ is completely uninformative about $Y$: $IG(Y|X)=0$
    - If $X$ is completely informative about $Y$: $IG(Y|X)=H(Y)$

[back to the top](#decision-tree)

### **4.** <big>Decision Tree Construction Algorithm</big>
- Simple, greedy, recursive approach, builds up tree node-by-node
  1. pick an attribute to split at a non-terminal node
  2. split examples into groups based on attribute value
  3. for each group:
    - if no examples - return majority from parent
    - else if all examples in same class - return class
    - else loop to step i.
- A good Tree
  - Not too small: need to handle important but possibly subtle distinctions indata
  - Not too big:
    - Computational efficiency (avoid redundant, spurious attributes)。
    - Avoid over-ftting training examples
  - Occam's Razor: fnd the simplest hypothesis (smallest tree) that fits theobservations
  - Inductive bias: small trees with informative nodes near the root

[back to the top](#decision-tree)

### **5.** <big>Selecting the right threshold</big>
- Step 1: Sort the Data
  - First, sort the data based on the values of the continuous attribute.
- Step 2: Calculate All Possible Thresholds
  - For the sorted data, thresholds can be taken as the midpoint between any twoconsecutive values. For example, if the sorted data is [1,2, 3, 4, 5], then thepossible thresholds are 1.5, 2.5, 3.5, and 4.5.
- Step 3: Calculate the Information Gain for Each Threshold 
  - For each possible threshold, split the data into two parts: values less than orequal to the threshold and values greater than the threshold. Then, calculate theInformation Gain for each split.
- Step 4: Choose the Threshold with Maximum Information Gain
  - From all possible thresholds, select the one with the maximum Information Gainas the fnal threshold.

[back to the top](#decision-tree)

### **6.** <big>Summary</big>
- Comparison to $k$-NN
  - $k$-NN:
    - Decision boundaries: piece-wise linear
    - Test complexity: non-parametric, few parameters besides training examples
  - Decision Trees:
    - Decision boundaries: axis-aligned, tree structured
    - Test complexity: attributes and splits
- Can express any Boolean function, but most useful when function dependscritically on few attributes
- Bad on:parity, majority functions; also not well-suited to continuousattributes 

[back to the top](#decision-tree)
