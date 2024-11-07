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
        self.init_weights()

    def init_weights(self):
        for i in range(len(self.layers) - 1):
            self.weights.append(np.random.randn(self.layers[i], self.layers[i + 1]) * np.sqrt(2 / self.layers[i]))
            # self.weights.append(np.random.randn(self.layers[i], self.layers[i + 1]) * 0.1)
            self.biases.append(np.zeros((1, self.layers[i + 1])))

    def sigmoid(self, x):
        # x = np.clip(x, -500, 500)  # 限制 x 的值以避免溢出
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
            if epoch % 100 == 0:
                loss = np.mean((self.forward(X) - y) ** 2)
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

# 绘制决策边界
def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    Z = np.round(model.predict(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8, cmap='viridis')
    plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), edgecolors='k', marker='o', cmap='viridis')
    plt.title('Decision Boundary')
    plt.show()


# 生成数据集
# X, y = utils.generate_dataset()
# X, y = make_moons(n_samples=1000, noise=0.5, random_state=42)
X = X_blobs
y = y_blobs
y = y.reshape(-1, 1)


# 训练和评估 MLP 使用不同的超参数
# layers_list = [[2, 10, 1], [2, 20, 1], [2, 10, 10, 1]]
layers_list = [[2, 10, 10, 1]]
best_score = 0
best_model = None
best_metrics = None

for layers in layers_list:
    model = MLP(layers)
    metrics = cross_validate(model, X, y, k=5, epochs=2000, learning_rate=0.1, batch_size=10, method='MBGD')
    if metrics['accuracy'] > best_score:
        best_score = metrics['accuracy']
        best_model = model
        best_metrics = metrics

# 展示结果
print(f"最佳模型层数: {best_model.layers}")
print(f"交叉验证得分: {best_metrics}")

# 绘制分类结果
plot_decision_boundary(best_model, X, y)
# plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), cmap='viridis')
# plt.title('数据集')
# plt.show()