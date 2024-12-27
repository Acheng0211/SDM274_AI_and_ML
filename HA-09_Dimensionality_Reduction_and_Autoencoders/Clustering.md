# Clustering
<div align="right">noted by Acheng0211(Guojing Huang, SUSTech)</div>

- [Clustering](#clustering)
    - [**1.** Unsupervised Learning](#1-unsupervised-learning)
    - [**2.** Clustering](#2-clustering)
    - [**3.** K-means](#3-k-means)
    - [**4.** *K*-means++](#4-k-means)
    - [**5.** Mini-Batch *K*-Means](#5-mini-batch-k-means)
    - [**6.** Soft *K*-Means](#6-soft-k-means)
    - [**7.** Lack of Clustering](#7-lack-of-clustering)
___


### **1.** <big>Unsupervised Learning</big>
- **Supervised learning**: produce desired outputs for given iuputs(inputs and targets are both given)
- Unsupervised learning algorithms have no explicit feedback whether outputs of system are correct(only inputs given, labels are unknown)
  - Tasks to consider:
    - Reduce dimensionality
    - Find clusters
    - Model data density
    - Find hidden causes
  - Key utility:
    - Compress data
    - Detect outliers
    - Facilitate other learning
  - Major types:
    - Dimensionality reduction: represent each input case using a smallnumber of variables (e.g., principal components analysis, factor analysis,independent components analysis)
    - Clustering: represent each input case using a prototype example (e.g..k-means, mixture models)
    - Density estimation: estimating the probability distribution over the dataspace

[back to the top](#clustering)

### **2.** <big>Clustering</big>
- Grouping *N* examples into *K* clusters
- Assume the data $\{x^{(1)}, \ldots, x^{(N)}\}$ lives in a Euclidean space, $x^{(n)} \in  \mathbb{R}^d$.
- Assume the data belongs to *K* classes (patterns)

[back to the top](#clustering)

### **3.** <big>K-means</big>
- Initialization: randomly initialize cluster centers
- The algorithm iteratively alternates between two steps:
    - Assignment step: Assign each data point to the closest cluster
    - Refitting step: Move each cluster center to the center of gravity of the。dataassigned to it
- Objective: Find cluster centers $m$ and assignments $r$ to minimize the sum of squared distanceof data points $\{x(n)\}$ to their assigned cluster centers
    $$\min_{\{m\}, \{r\}} J(\{m\}, \{r\}) = \min_{\{m\}, \{r\}} \sum_{n=1}^{N} \sum_{k=1}^{K} r_k^{(n)} \| m_k - x^{(n)} \|^2
    $$
    $$\text{s.t.} \sum_{k} r_k^{(n)} = 1, \forall n, \quad \text{where} \quad r_k^{(n)} \in \{0, 1\}, \forall k, n
    $$
where $ r_k^{(n)} = 1 $ means that $ x^{(n)} $ is assigned to cluster $ k $ with center $ m_k $.
- **Optimization method** is a form of coordinate descent ("block coordinatedescent")
    - Fix centers, optimize assignments (choose cluster whose mean is closest)
    - Fix assignments, optimize means (average of assigned datapoints)
- Algorithm
  - Initialization: Set K cluster means $m_1, \ldots , mk$ to random values
  - Repeat until convergence (until assignments do not change):
  1. Assignment: Each data point $x^{(n)}$ assigned to nearest mean $\hat k^n= \displaystyle \arg\min_k d(m_k,x^{(n)})$
and Responsibilities (1 of k encoding) $r^{(n)}_k = 1 \leftrightarrow \hat k^{(n)}=k$
  2. Update: Model parameters, means are adjusted to match sample means ofdata points they are responsible for: $m_k = \frac{\sum_n r^{(n)}_k x^{(n)}}{\sum_n r^{(n)}_k }$
- assignment is changed / a cluster center is moved $\rightarrow$ the sum sqaured distances $J$ of data points from their assigned cluster centers is reduced. converge when no change in the assignment step. The objective $j$ is non-convex (so coordinate decent on $J$ is not guaranteed to converge to the global minimum)
- Stuck at local minima is unavoidable for k-means, but could try:
  - many random starting points
  - non-local split-and-merge moves
    - Simultaneously merge two nearby clusters
    - and split a big cluster into two

[back to the top](#clustering)

### **4.** <big>*K*-means++</big>
- improvement over K-means
  - impove the initial placement of cluster centers, leading to better quality clusters and faster convergence
  - reduce the likelihood of getting stuck in poor local minima
- Algorithm
  - Initialize One Center:
    - Choose the first center point $m_1$, randomly from the data points.
    - For each data point $x^{(n)}$, calculate the distance $D(x^{(n)})$ to the nearestcenter point that has already been chosen.
  - Select the Next Center (Farthest point):
    - For each data point $x^{(n)}$ , calculate the probability $p(x^{(n)}) = \frac{D(x^{(n)})^2}{\sum_j D(x^{(j)})^2}$ 
    - Choose the next center mk+1 from the data points according to p(æ(”))
  - Repeat Step 2 until the desired number of centers K is reached.
  - Run *K*-means algorithm
- Get optimal *K*
  - A commonly used method is to test different $K$ and measures the resulting **SSE** $J$. 
  - The value of Kis chosen where an **increase leads to a very small decrease** in SSE, and a d**ecrease leads to a sharp increase**
  - This point, which defines the optimal K, is known as the **"elbow point"**
  
[back to the top](#clustering)

### **5.** <big>Mini-Batch *K*-Means</big>
- Mini-Batch K-Means is a variant of the -Means algorithm that is moreefficient for large datasets by processing a small random sample of thedata at each iteration instead of the entire dataset.
1. Initialize Centers:
    - Randomly select a subset of data points, called a mini-batcho - Initialize cluster centers from the mini-batch.
2. Assign Clusters:
    - For each data point in the mini-batch, assign it to the nearest center.
3. Update Centers:
    - Update the positions of the centers by calculating the mean of the points assigned to each cluster in the mini-batch.
4.  Repeat Steps 2 and 3:
    - Repeat steps 2 and 3 for a fixed number of iterations or until the centersconverge (i.e., the change in the positions of the centers is below acertain threshold).
5. Expand Mini-Batch Size (optional):
    - Optionally, increase the size of the mini-batch over time to improve theaccuracy of the center updates.
6. Refnement:
    - After the initial convergence with mini-batches, the algorithm can berefned by running a few iterations of the standard K-Means algorithmon the entire dataset
7. Convergence:
    - The algorithm converges when the assignments of the data points to theclusters do not change significantly, or the change in the within-clustersum of squares is below a certain threshold.
- Mini-Batch K-Means is particularly useful for large-scale clustering taskswhere processing the entire dataset at once is computationally expensive orimpractical.
  
[back to the top](#clustering)

### **6.** <big>Soft *K*-Means</big>
- soft assignment: instead of making hard assignments of data points to clusters, allows a cluster to use more information about the data in the refitting step.
- Algorithm
  - **Initialization**: Set $ K $ means $\{m_k\}$ to random values.
  - Repeat until convergence (until assignments do not change):

1. **Assignment**: Each data point $ n $ given soft "degree of assignment" to each cluster mean $ k $, based on responsibilities
   $$ r_k^{(n)} = \frac{\exp[-\beta d(m_k, x^{(n)})]}{\sum_j \exp[-\beta d(m_j, x^{(n)})]}
   $$

2. **Update**: Model parameters, means, are adjusted to match sample means of data points they are responsible for:
   $$ m_k = \frac{\sum_n r_k^{(n)} x^{(n)}}{\sum_n r_k^{(n)}}
   $$

[back to the top](#clustering)

### **7.** <big>Lack of Clustering</big>
- Perform poorly on:
  - anisotropic data
  - data with size variation

[back to the top](#clustering)