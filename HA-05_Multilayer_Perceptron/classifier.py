import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

class MLP:
    def __init__(self, layers):
        self.layers = layers
        self.weights = []
        self.biases = []
        self.initialize_weights()

    def initialize_weights(self):
        for i in range(len(self.layers) - 1):
            weight = np.random.randn(self.layers[i], self.layers[i + 1]) * 0.1
            bias = np.zeros((1, self.layers[i + 1]))
            self.weights.append(weight)
            self.biases.append(bias)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        self.a = [X]
        for i in range(len(self.weights)):
            net = np.dot(self.a[-1], self.weights[i]) + self.biases[i]
            out = self.sigmoid(net)
            self.a.append(out)
        return self.a[-1]

    def backward(self, X, y, learning_rate):
        m = y.shape[0]
        deltas = [self.a[-1] - y]
        for i in range(len(self.weights) - 1, 0, -1):
            delta = np.dot(deltas[-1], self.weights[i].T) * self.sigmoid_derivative(self.a[i])
            deltas.append(delta)
        deltas.reverse()

        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * np.dot(self.a[i].T, deltas[i]) / m
            self.biases[i] -= learning_rate * np.mean(deltas[i], axis=0, keepdims=True)

    def train(self, X, y, epochs, learning_rate, batch_size=None):
        if batch_size is None:
            batch_size = X.shape[0]
        for epoch in range(epochs):
            for i in range(0, X.shape[0], batch_size):
                X_batch = X[i:i + batch_size]
                y_batch = y[i:i + batch_size]
                self.forward(X_batch)
                self.backward(X_batch, y_batch, learning_rate)
            if epoch % 100 == 0:
                loss = np.mean((self.forward(X) - y) ** 2)
                print(f'Epoch {epoch}, Loss: {loss}')
    
    def predict(self, X):
        return self.forward(X)

def cross_validate(model, X, y, k=5, epochs=100, learning_rate=0.01, batch_size=None):
    kf = KFold(n_splits=k)
    scores = {'accuracy': [], 'recall': [], 'precision': [], 'f1': []}
    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        model.train(X_train, y_train, epochs, learning_rate, batch_size)
        predictions = np.round(model.predict(X_val))
        scores['accuracy'].append(accuracy_score(y_val, predictions))
        scores['recall'].append(recall_score(y_val, predictions, average='weighted'))
        scores['precision'].append(precision_score(y_val, predictions, average='weighted'))
        scores['f1'].append(f1_score(y_val, predictions, average='weighted'))
    return {metric: np.mean(values) for metric, values in scores.items()}

def generate_classification_data(n_samples=100):
    np.random.seed(0)
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 < 1).astype(int).reshape(-1, 1)
    return X, y

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
        plt.plot(X, model.predict(X), label=f'Model {layers}')
        plt.contourf(xx, yy, Z, alpha=0.8, cmap='viridis')
        plt.title(f'MLP_Decision_Boundary_{gd}')
        plt.legend()
        # plt.savefig(os.path.join('./output', f'MLP_Decision_Boundary_{gd}_{layers}.png'))
        plt.show()

# 示例用法
if __name__ == "__main__":
    # 创建一个MLP实例，输入层2个单元，隐藏层10个单元，输出层1个单元
    layers_list = [[2,10,1]]
    mlp = MLP(layers_list)
    # 生成一些分类数据
    # X, y = generate_classification_data()
    X, y = make_moons(n_samples=100, noise=0.2, random_state=42)
    y = y.reshape(-1, 1)
    # 交叉验证模型
    scores = cross_validate(mlp, X, y, k=5, epochs=8000, learning_rate=0.01)
    print(f'Cross-validation scores: {scores}')

    # 可视化分类结果
    plot_decision_boundary(X, y, mlp, layers_list, "MBGD")
