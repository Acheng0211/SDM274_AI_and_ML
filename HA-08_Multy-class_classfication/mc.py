from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import os

def load_data(train_file, test_file):
    train_data = pd.read_csv(train_file, header=None)
    X_train = train_data.iloc[:, :-1].values
    y_train = train_data.iloc[:, -1].values
    test_data = pd.read_csv(test_file, header=None)
    X_test = test_data.iloc[:, :-1].values
    y_test = test_data.iloc[:, -1].values

    X_train = X_train / 16.0
    X_test = X_test / 16.0

    encoder = OneHotEncoder(sparse=False)
    y_train = encoder.fit_transform(y_train.reshape(-1, 1))
    y_test = encoder.transform(y_test.reshape(-1, 1))

    return X_train, y_train, X_test, y_test

def plot_loss(model):
    loss = model.losses
    plt.plot(loss)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'multi-class classification loss')
    plt.savefig(os.path.join('./HA-08_Multy-class_classfication/output', f'multi-class_classification_loss.png'))
    plt.show()

class MLP:
    def __init__(self, layers, learning_rate=0.01, epochs=1000):
        self.layers = layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = []
        self.biases = []
        self.losses = []
        self.init_weights()

    def init_weights(self):
        for i in range(len(self.layers) - 1):
            self.weights.append(np.random.randn(self.layers[i], self.layers[i + 1]) * np.sqrt(2 / self.layers[i]))
            self.biases.append(np.zeros((1, self.layers[i + 1])))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward_propagation(self, X):
        self.a = [X]
        for i in range(len(self.weights)):
            net = np.dot(self.a[-1], self.weights[i]) + self.biases[i]
            self.a.append(self.sigmoid(net))
        return self.a[-1]

    def backward_propagation(self, X, y):
        m = y.shape[0]
        self.deltas = [self.a[-1] - y]
        for i in range(len(self.a) - 2, 0, -1):
            delta = np.dot(self.deltas[-1], self.weights[i].T) * self.sigmoid_derivative(self.a[i])
            self.deltas.append(delta)
        self.deltas.reverse()
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * np.dot(self.a[i].T, self.deltas[i]) / m
            self.biases[i] -= self.learning_rate * np.mean(self.deltas[i], axis=0, keepdims=True)

    def fit(self, X, y):
        for epoch in range(self.epochs):
            y_pred = self.forward_propagation(X)
            self.backward_propagation(X, y)
            # loss = np.mean((self.forward_propagation(X) - y) ** 2)
            loss = self._cross_entropy_loss(y, y_pred)
            self.losses.append(loss)
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')

    def _cross_entropy_loss(self, y, y_pred):
        epsilon = 0
        return -np.mean(y * np.log(y_pred + epsilon) + (1-y) * np.log(1-y_pred + epsilon))

    def predict(self, X):
        return self.forward_propagation(X)
    
    def evaluate(self, X, y):
        predictions = self.predict(X)
        predicted_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(y, axis=1)

        accuracy = np.mean(predicted_classes == true_classes)
        precision = np.zeros(10)
        recall = np.zeros(10)
        f1 = np.zeros(10)

        for i in range(10):
            tp = np.sum((predicted_classes == i) & (true_classes == i))
            fp = np.sum((predicted_classes == i) & (true_classes != i))
            fn = np.sum((predicted_classes != i) & (true_classes == i))

            precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0

        precision = np.mean(precision)
        recall = np.mean(recall)
        f1 = np.mean(f1)

        evaluation = {'accuracy': [accuracy], 'recall': [recall], 'precision': [precision], 'f1': [f1]}
        return {metric: np.mean(values) for metric, values in evaluation.items()}

if __name__ == "__main__":
    train_file = './HA-08_Multy-class_classfication/data/optdigits.tra'
    test_file = './HA-08_Multy-class_classfication/data/optdigits.tes'
    X_train, y_train, X_test, y_test = load_data(train_file, test_file)

    input_size = X_train.shape[1]
    layers = [input_size, 32, 10]
    model = MLP(layers, learning_rate=0.01, epochs=10000)
    model.fit(X_train, y_train)

    metrics = model.evaluate(X_test, y_test)
    print(f"MC evaluation: {metrics}")
    plot_loss(model)
