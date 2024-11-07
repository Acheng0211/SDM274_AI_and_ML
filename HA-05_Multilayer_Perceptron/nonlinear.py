import numpy as np 
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

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


def cross_validate(model, X, y, k=5, epochs=100, learning_rate=0.01, batch_size=None):
    kf = KFold(n_splits=k)
    scores = []
    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]
        model.train(X_train, y_train, epochs, learning_rate, batch_size)
        predictions = model.predict(X_val)
        loss = mean_squared_error(y_val, predictions)
        scores.append(loss)
        # accuracy = np.mean(np.round(predictions) == y_val)
        # scores.append(accuracy)
    return np.mean(scores)

# 生成数据集
def nonlinear_function(x):
    return np.sin(2 * np.pi * x) + 0.1 * np.random.randn(*x.shape)

if __name__ == "__main__":
    X = np.linspace(0, 1, 100).reshape(-1, 1)
    y = nonlinear_function(X)

    # 训练和评估 MLP 使用不同的超参数
    layers_list = [[1, 10, 1], [1, 20, 1], [1, 5, 10, 1]]
    best_score = 0
    best_model = None

    best_model = MLP([1, 10, 1])
    best_score = cross_validate(best_model, X, y, k=5, epochs=3000, learning_rate=0.1, batch_size=10)

    # for layers in layers_list:
    #     model = MLP(layers)
    #     score = cross_validate(model, X, y, k=5, epochs=10000, learning_rate=0.01, batch_size=10)
    #     print(f"结果1: {layers}, 交叉验证得分: {score}")
    #     if score > best_score:
    #         best_score = score
    #         best_model = model

    # 绘制结果
    plt.scatter(X, y, label='raw data')
    plt.plot(X, best_model.predict(X), label='MLP approximation', color='red')
    plt.legend()
    plt.show()

    print(f"最佳模型层数: {best_model.layers}, 交叉验证得分: {best_score}")