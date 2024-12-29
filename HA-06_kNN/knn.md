# Nearest Neighbors 
<div align="right">noted by Acheng0211(Guojing Huang, SUSTech)</div>

- [Nearest Neighbors](#nearest-neighbors)
    - [**1.** Non-parametric Models](#1-non-parametric-models)
    - [**2.** Nearest Neighbors](#2-nearest-neighbors)
    - [**3.** $k$-Nearest Neighbors](#3-k-nearest-neighbors)
    - [**4.** KD Tree](#4-kd-tree)
    - [**5.** Summary](#5-summary)
___


### **1.** <big>Non-parametric Models</big>
- Classification is intrinsically **non-linear**

[back to the top](#nearest-neighbors)

### **2.** <big>Nearest Neighbors</big>
- Training example in Euclidean space: $x \in R^d$
- Distance typically defined to be Euclidean:$${\|x^{(a)} - x^{(b)}\|}_2 = \sqrt{\displaystyle \sum^d_{j=1}(x^{(a)}_j - x^{(b)}_j)^2}$$
- Decision boundaries
    - NN does not explicitly compute decision boundaries but can be inferred
    - **Voronoi diagram**

[back to the top](#nearest-neighbors)

### **3.** <big>$k$-Nearest Neighbors</big>
- smooth when disturbed by mis-labeled data("class noise")
- Algorithm
  - Find $k$ examples {**x**$^{(i)}, t^{(i)}$} close to the test instance **x**
  - Classification output is majority class
  $$y = arg \displaystyle max_{t^{(z)}} \sum^k_{r=1}\delta(t^{(z)}, t^{(r)})$$
- Rules to choose $k$
  - larger $k$ may lead to better performance
  - but too large $k$ may be far away from the quety
  - use cross-validation to find $k$
  - Rule of tumb is $k < \sqrt{n}$, where $n$ is the number of training examples
- Issues & Remedies
  - some attributes (coordinates of $x$) with larger ranges will be treated as more important, which nedd to **nomalize scale**: [0,1] or $(x_j - m)/\sigma$
  - Irrelevant, correlated attributes add noise to distance measure
    - eliminate some attributes
    - vary and possibly adapt weight of attributes
  - Non-metric attributes(symbols)
    - Hamming distance
  - Expensive at test time: To find one nearest neighbor of a query point $x$, we must compute the distance to all N training examples. **Complexity: O(kdN) for $k$-NN**
    - Use subset of dimensions
    - Pre-sort training examples into fast data structures (e.g., kd-treees)
    - Compute only an approximate distance (e.g., LSH)
    - Remove redundant data (e.g., condensing).
  - Storage Requirements: Must store all training data
    - Remove redundant data (e.g., condensing).
    - Pre-sorting often increases the storage requirements
  - High Dimensional Data: "Curse of Dimensionality"
    - Required amount of training data increases exponentially with diimension
    - Computational cost also increases

[back to the top](#nearest-neighbors)

### **4.** <big>KD Tree</big>
- KD tree(*K*-Dimensional Tree)
- Definition
  - Node: Each node in a KD tree represents a point in K-dimensional space
  - Axis Selection: During the construction of the tree, a dimension (axis) is
  chosen each time and points are partitioned based on that dimernsion's value.
  - Partitioning: A value is chosen along the selected axis to spplit the points into two parts: one part with points having values less thain the chosen value on that axis, and the other part with points having valuesgreater than or equal to the chosen value.
  - Recursive Construction: The same process is applied recursively to each subset until each subset contains only one point or no points.
  1. Choose a dimension (axis).
  2. Sort the points along that dimension.
  3. Select the median as the partition point, dividing the pointsinto two groups.
  4. Recursively apply the above steps to each subset until eaclh subset contains
  only one point or no points.

[back to the top](#nearest-neighbors)

### **5.** <big>Summary</big>
- $k$-NN naturally forms complex decision boundaries and adapts to data density
- $k$-NN typically works well with lots of samples
- Problems:
  - Sensitive to class noise
  - Sensitive to scales of attributes
  - Distances are less meaningful in high dimensions
  - Scales linearly with number of examples
  
[back to the top](#nearest-neighbors)