import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import os

def generate_data(mission):
    if mission == "Classifier":
        X, y = make_moons(n_samples=500, noise=0.3, random_state=42)
        y = y.reshape(-1, 1)
        y_raw = 0
    elif mission == "Nonlinear":
        X = np.linspace(0, 1, 100).reshape(-1, 1)
        y_raw = np.sin(2 * np.pi * X) + 0.1 * np.random.randn(*X.shape)
        y = np.round(y_raw) # ensure y is discrete
    return X, y, y_raw

def generate_dataset():
    np.random.seed(3407)
    class_1 = np.hstack([np.random.normal( 1, 1, size=(500, 2)),  np.ones(shape=(500, 1))])
    class_2 = np.hstack([np.random.normal(-1, 1, size=(500, 2)), -np.ones(shape=(500, 1))])
    dataset = np.vstack([class_1, class_2])
    X, y = dataset[:,:2], dataset[:,2]
    return X, y

def load_data(file_name):
    # 读取数据集
    column_names = ['UDI'] + [f'feature_{i}' for i in range(13)]  
    data_raw = pd.read_csv(file_name, header=None, names=column_names)

    return data_raw

def filter(data):
    # 删除 class_label = 3 的所有行
    data_filtered = data[data['class_label'] != 3]
    return data_filtered

def classify_data(data):
    # 将数据分为特征和标签
    X = data.drop('class_label', axis=1).values #data_filtered
    y = data['class_label'].values

    return X, y

def load_and_process_data(file_name, features_to_remove=None):
    # read the data
    data_raw = pd.read_csv(file_name)
    # handle missing values using the mean of the column
    #data_raw.fillna(data_raw.mean(), inplace=True)

    # extract features and target variable
    features = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
    target = 'Machine failure'
    X = data_raw[features].values
    y = data_raw[target].values

    # apply delete features option
    if features_to_remove:
        X = np.delete(X, features_to_remove, axis=1)

    return X, y.reshape(-1,1)

def split_data(X, y, test_size=0.3, val_size = 0.2, random_state = 42):
    # 划分训练集和测试集，这里我们按照70%训练集，30%测试集的比例来划分
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=random_state)
    #Perceptron时-1, LogisticRegression时0
    # y_train[y_train == 2] = -1
    # y_test[y_test == 2] = -1
    # y_train[y_train == 2] = 0
    # y_test[y_test == 2] = 0
    # y_train = y_train.reshape(-1,1)
    # y_test = y_test.reshape(-1,1)

    # 打印结果以确认划分
    # print("Training set shape:", X_train.shape)
    # print("Test set shape:", X_test.shape)

    return X_train, X_test, y_train.reshape(-1,1), y_test.reshape(-1,1)

def cross_validate(model, X, y, k=5, epochs=100, learning_rate=0.01, batch_size=None, gd='MBGD'):
    kf = KFold(n_splits=k)
    scores = {'accuracy': [], 'recall': [], 'precision': [], 'f1': []}
    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        model.train(X_train, y_train, epochs, learning_rate, batch_size, gd)
        predictions = np.round(model.predict(X_val))
        scores['accuracy'].append(accuracy_score(y_val, predictions))
        scores['recall'].append(recall_score(y_val, predictions, average='weighted', zero_division=0))
        scores['precision'].append(precision_score(y_val, predictions, average='weighted', zero_division=0))
        scores['f1'].append(f1_score(y_val, predictions, average='weighted', zero_division=0))
    return {metric: np.mean(values) for metric, values in scores.items()}

def plot_nonliear(X, y, models, layers_list, gd):
    plt.figure()
    plt.scatter(X, y, label='raw data')
    for model, layers in zip(models, layers_list):
        plt.plot(X, model.predict(X), label=f'Model {layers}')
    plt.title(f'MLP_Nonlinear_Approximation_{gd}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.savefig(os.path.join('./output', f'MLP_Nonlinear_Approximation_{gd}.png'))
    plt.show()

def plot_decision_boundary(X, y, models, layers_list, gd):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))   
    for model, layers in zip(models, layers_list):
        plt.figure()
        plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), edgecolors='k', marker='o', cmap='viridis')
        Z = np.round(model.predict(np.c_[xx.ravel(), yy.ravel()]))
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.8, cmap='viridis', label=f'Model {layers}')
        plt.title(f'MLP_Decision_Boundary_{gd}')
        plt.legend()
        # plt.savefig(os.path.join('./output', f'MLP_Decision_Boundary_{gd}_{layers}.png'))
        plt.show()

# def plot_decision_boundary(X, y, model, layers, gd):
#     x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#     y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#     xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
#                          np.arange(y_min, y_max, 0.01))
#     Z = np.round(model.predict(np.c_[xx.ravel(), yy.ravel()]))
#     Z = Z.reshape(xx.shape)   
#     plt.figure()
#     plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), edgecolors='k', marker='o', cmap='viridis')
#     # plt.plot(X, model.predict(X), label=f'Model {layers}')
#     plt.contourf(xx, yy, Z, alpha=0.8, cmap='viridis')
#     plt.title(f'MLP_Decision_Boundary_{gd}')
#     plt.legend()
#     plt.savefig(os.path.join('./output', f'MLP_Decision_Boundary_{gd}_{layers}.png'))
#     plt.show()

def plot_loss(models, layers_list, mission, gd):
    plt.figure(figsize=(10, 6))
    for model, layers in zip(models, layers_list):
        plt.plot(model.loss, label=f'Model {layers}')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    if mission == "Nonlinear":
        plt.title(f'MLP_Nonlinear_Approximation_loss_{gd}')
        plt.savefig(os.path.join('./output', f'MLP_Nonlinear_Approximation_loss_{gd}.png'))
    elif mission == "Classifier":
        plt.title(f'MLP_Classifier_loss_{gd}')
        plt.savefig(os.path.join('./output', f'MLP_Classifier_loss_{gd}.png'))
    plt.show()

def LOG_RESULTS(results, gd):
    print(f"Update method: {gd}")
    for layers, metrics in results:
        print(f"model: {layers}, evaluation: {metrics}")
    