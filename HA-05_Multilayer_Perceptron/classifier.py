import numpy as np 
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import utils
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_blobs, make_gaussian_quantiles

# 生成方形数据集
X_square, y_square = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=42)

# 生成多个簇的数据集
X_blobs, y_blobs = make_blobs(n_samples=1000, centers=4, cluster_std=1.0, random_state=42)

# 生成高斯分布数据集
X_gaussian, y_gaussian = make_gaussian_quantiles(n_samples=1000, n_features=2, n_classes=2, random_state=42)

class MLP:
    def __init__(self, layers):
        self.layers = layers
        self.weights = []
        self.biases = []
        self.loss = []
        self.init_weights()

    def init_weights(self):
        for i in range(len(self.layers) - 1):
            self.weights.append(np.random.randn(self.layers[i], self.layers[i + 1]) * np.sqrt(2 / self.layers[i]))
            # self.weights.append(np.random.randn(self.layers[i], self.layers[i + 1]) * 0.1)
            self.biases.append(np.zeros((1, self.layers[i + 1])))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        self.a = [X]
        for i in range(len(self.weights) - 1):
            net = np.dot(self.a[-1], self.weights[i]) + self.biases[i]
            self.a.append(self.sigmoid(net))
        # 最后一层不使用激活函数
        net = np.dot(self.a[-1], self.weights[-1]) + self.biases[-1]
        self.a.append(net)
        return self.a[-1]

    def backward(self, X, y, learning_rate):
        m = y.shape[0]
        self.deltas = [self.a[-1] - y]
        for i in range(len(self.a) - 2, 0, -1):
            delta = np.dot(self.deltas[-1], self.weights[i].T) * self.sigmoid_derivative(self.a[i])
            self.deltas.append(delta)
        self.deltas.reverse()
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * np.dot(self.a[i].T, self.deltas[i]) / m
            self.biases[i] -= learning_rate * np.sum(self.deltas[i], axis=0, keepdims=True) / m

    def train(self, X, y, epochs, learning_rate, batch_size=None, method='SGD'):
        self.gd = method
        if method not in ['SGD', 'MBGD']:
            raise ValueError("method should be either 'SGD' or 'MBGD'")
        
        if method == 'SGD':
            batch_size = 1
        elif method == 'MBGD' and batch_size is None:
            batch_size = X.shape[0]
        
        for epoch in range(epochs):
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            for start in range(0, X.shape[0], batch_size):
                end = start + batch_size
                X_batch, y_batch = X[indices[start:end]], y[indices[start:end]]
                self.forward(X_batch)
                self.backward(X_batch, y_batch, learning_rate)
            loss = np.mean((self.forward(X) - y) ** 2)
            self.loss.append(loss)
            if epoch % 100 == 0:            
                print(f'Epoch {epoch}, Loss: {loss}')

    def predict(self, X):
        return self.forward(X)

def cross_validate(model, X, y, k=5, epochs=100, learning_rate=0.01, batch_size=None, method='MBGD'):
    kf = KFold(n_splits=k)
    scores = {'accuracy': [], 'recall': [], 'precision': [], 'f1': []}
    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        model.train(X_train, y_train, epochs, learning_rate, batch_size, method)
        predictions = np.round(model.predict(X_val))
        scores['accuracy'].append(accuracy_score(y_val, predictions))
        scores['recall'].append(recall_score(y_val, predictions, average='weighted'))
        scores['precision'].append(precision_score(y_val, predictions, average='weighted'))
        scores['f1'].append(f1_score(y_val, predictions, average='weighted'))
    return {metric: np.mean(values) for metric, values in scores.items()}


if __name__ == "__main__":
    # X, y = utils.generate_dataset()
    # np.random.seed(0)
    # X = np.random.randn(100, 2)
    # y = (X[:, 0] ** 2 + X[:, 1] ** 2 < 1).astype(int).reshape(-1, 1)
    X, y = make_moons(n_samples=100, noise=0.2, random_state=42)
    y = y.reshape(-1, 1)

    # 训练和评估 MLP 使用不同的超参数
    layers_list = [[2, 10, 1],[2, 20, 1],[2, 10, 20, 1]]
    layers = [2, 10, 1]
    models = []
    results = []
    best_score = 0
    best_model = None
    method = "SGD"
    
    for layers in layers_list:
        model = MLP(layers)
        metrics = cross_validate(model, X, y, k=5, epochs=8000, learning_rate=0.01, batch_size=10, method=method)
        print(f"layer:{layers} train complete")
        utils.plot_decision_boundary(X, y, model, layers, method)
        models.append(model)
        results.append((layers, metrics))
        
    # 绘制结果
    utils.plot_loss(models, layers_list, "Classifier", method)

    utils.LOG_RESULTS(results,method)