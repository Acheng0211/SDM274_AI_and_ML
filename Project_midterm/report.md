# AI and Machine Leanring Midterm Project Report
<div align="right">12111820 黄国靖</div>

## Tasks:
1. **Data Preprocessing:**
   - Load the dataset and perform basic data cleaning to handle missing values and irrelevant features.
   - Normalize or standardize the features if necessary.

2. **Feature Engineering:**
   - Select or engineer features that could be relevant for predicting machine failures.

3. **Model Implementation:**
   - Implement linear regression, perceptron, and logistic regression models to predict machine failures.
   - Implement an MLP model to predict machine failures, using at least one hidden layer.

4. **Model Training and Evaluation:**
   - Split the dataset into training and testing sets (e.g., 70% training, 30% testing).
   - Train each model on the training set and evaluate their performance on the testing set.
   - Use appropriate metrics such as accuracy, precision, recall, F1-score, to evaluate the models.

5. **Model Comparison:**
   - Compare the performance of the different models.
   - Discuss the strengths and weaknesses of each model in the context of predictive maintenance.

6. **Hyperparameter Tuning:**
   - Perform hyperparameter tuning for the perceptron, logistic regression, and MLP models to improve their performance.

## Methodology
1. Load the dataset, using mean method to handle missing values and extract relevant features
2. Split the dataset into 70% training and 30% testing sets
3. Implement linear regression, perceptron, logistic regression and MLP models
4. Compare the evaluation of 4 models using all the features
5. Compare the evaluation of logistic regression model under the conditions of diffenrent input features

## Findings and Conclusions

![参数配置](./output/settings.png "参数配置") 

## Results
#### Task1: Compare the evaluation of 4 models using all the features
```python
Linear Regression evaluation:{'accuracy': 0.031, 'recall': 0.031, 'precision': 0.000961, 'f1': 0.0018642095053346267}
Perceptron evaluation: {'accuracy': 0.031, 'recall': 0.031, 'precision': 0.000961, 'f1': 0.0018642095053346267}
Logistic Regression evaluation: {'accuracy': 0.969, 'recall': 0.969, 'precision': 0.9389609999999999, 'f1': 0.9537440325038092} 
MLP evaluation: {'accuracy': 0.9661, 'recall': 0.9661, 'precision': 0.9338183500000001, 'f1': 0.9495677504357161}
```

#### Task2: Compare the evaluation of logistic regression model under the conditions of diffenrent input features
```python
Logistic Regression evaluation with all features: {'accuracy': 0.969, 'recall': 0.969, 'precision': 0.9389609999999999, 'f1': 0.9537440325038092}
Logistic Regression evaluation without two temperature features: {'accuracy': 0.8946666666666667, 'recall': 0.8946666666666667, 'precision': 0.9553094565487982, 'f1': 0.9209330197885173}
Logistic Regression evaluation with only two temperature features: {'accuracy': 0.969, 'recall': 0.969, 'precision': 0.9389609999999999, 'f1': 0.9537440325038092}
```

## Findings 
1. Mini-Batch update is better when network is bigger, in contrast of better performance while smaller network using stochastic update.
2. As all the networks in this assignment are constructed with small numbers of layers and the number of inputs is also small, the network with more neurons per layer and less number of layers somehow occur with the best performance in both tasks.

## Conclusions