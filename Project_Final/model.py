import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.name = 'PCA'

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X = X - self.mean
        cov = np.cov(X.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        eigenvectors = eigenvectors.T
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]
        self.components = eigenvectors[:self.n_components]

    def transform(self, X):
        X = X - self.mean
        return np.dot(X, self.components.T)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

class NonLinearAutoencoder1:
    def __init__(self, input_dim, encoding_dim, hidden_dim, learning_rate=0.01, epochs=2000):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = {
            'encoder_hidden': np.random.randn(input_dim, hidden_dim),
            'hidden_encoding': np.random.randn(hidden_dim, encoding_dim),
            'encoding_hidden': np.random.randn(encoding_dim, hidden_dim),
            'hidden_decoder': np.random.randn(hidden_dim, input_dim)
        }
        self.biases = {
            'encoder_hidden': np.zeros((1, hidden_dim)),
            'hidden_encoding': np.zeros((1, encoding_dim)),
            'encoding_hidden': np.zeros((1, hidden_dim)),
            'hidden_decoder': np.zeros((1, input_dim))
        }
        self.losses = []

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def fit(self, X):
        for epoch in range(self.epochs):
            hidden = self.relu(np.dot(X, self.weights['encoder_hidden']) + self.biases['encoder_hidden'])
            encoded = self.relu(np.dot(hidden, self.weights['hidden_encoding']) + self.biases['hidden_encoding'])
            hidden_decoded = self.relu(np.dot(encoded, self.weights['encoding_hidden']) + self.biases['encoding_hidden'])
            decoded = self.relu(np.dot(hidden_decoded, self.weights['hidden_decoder']) + self.biases['hidden_decoder'])
            loss = np.mean((X - decoded) ** 2)
            self.losses.append(loss)
            error = X - decoded
            d_hidden_decoder = error * self.relu_derivative(decoded)
            d_encoding_hidden = np.dot(d_hidden_decoder, self.weights['hidden_decoder'].T) * self.relu_derivative(hidden_decoded)
            d_hidden_encoding = np.dot(d_encoding_hidden, self.weights['encoding_hidden'].T) * self.relu_derivative(encoded)
            d_encoder_hidden = np.dot(d_hidden_encoding, self.weights['hidden_encoding'].T) * self.relu_derivative(hidden)
            self.weights['hidden_decoder'] += self.learning_rate * np.dot(hidden_decoded.T, d_hidden_decoder)
            self.biases['hidden_decoder'] += self.learning_rate * np.sum(d_hidden_decoder, axis=0, keepdims=True)
            self.weights['encoding_hidden'] += self.learning_rate * np.dot(encoded.T, d_encoding_hidden)
            self.biases['encoding_hidden'] += self.learning_rate * np.sum(d_encoding_hidden, axis=0, keepdims=True)
            self.weights['hidden_encoding'] += self.learning_rate * np.dot(hidden.T, d_hidden_encoding)
            self.biases['hidden_encoding'] += self.learning_rate * np.sum(d_hidden_encoding, axis=0, keepdims=True)
            self.weights['encoder_hidden'] += self.learning_rate * np.dot(X.T, d_encoder_hidden)
            self.biases['encoder_hidden'] += self.learning_rate * np.sum(d_encoder_hidden, axis=0, keepdims=True)
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')

    def encode(self, X):
        hidden = self.relu(np.dot(X, self.weights['encoder_hidden']) + self.biases['encoder_hidden'])
        return self.relu(np.dot(hidden, self.weights['hidden_encoding']) + self.biases['hidden_encoding'])

    def decode(self, encoded):
        hidden_decoded = self.relu(np.dot(encoded, self.weights['encoding_hidden']) + self.biases['encoding_hidden'])
        return self.relu(np.dot(hidden_decoded, self.weights['hidden_decoder']) + self.biases['hidden_decoder'])

    def reconstruct(self, X):
        encoded = self.encode(X)
        decoded = self.decode(encoded)
        return decoded

class NonLinearAutoencoder:
    def __init__(self, input_dim, encoding_dim, hidden_dim, learning_rate=0.01, epochs=1000):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = {
            'encoder_hidden': np.random.randn(input_dim, hidden_dim),
            'hidden_encoding': np.random.randn(hidden_dim, encoding_dim),
            'encoding_hidden': np.random.randn(encoding_dim, hidden_dim),
            'hidden_decoder': np.random.randn(hidden_dim, input_dim)
        }
        self.biases = {
            'encoder_hidden': np.zeros((1, hidden_dim)),
            'hidden_encoding': np.zeros((1, encoding_dim)),
            'encoding_hidden': np.zeros((1, hidden_dim)),
            'hidden_decoder': np.zeros((1, input_dim))
        }
        self.losses = []
        self.name = 'NonLinearAutoencoder'

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def fit(self, X):
        for epoch in range(self.epochs):
            hidden = self.sigmoid(np.dot(X, self.weights['encoder_hidden']) + self.biases['encoder_hidden'])
            encoded = self.sigmoid(np.dot(hidden, self.weights['hidden_encoding']) + self.biases['hidden_encoding'])
            hidden_decoded = self.sigmoid(np.dot(encoded, self.weights['encoding_hidden']) + self.biases['encoding_hidden'])
            decoded = self.sigmoid(np.dot(hidden_decoded, self.weights['hidden_decoder']) + self.biases['hidden_decoder'])
            
            loss = np.mean((X - decoded) ** 2)
            self.losses.append(loss)
            
            error = X - decoded
            d_hidden_decoder = error * self.sigmoid_derivative(decoded)
            d_encoding_hidden = np.dot(d_hidden_decoder, self.weights['hidden_decoder'].T) * self.sigmoid_derivative(hidden_decoded)
            d_hidden_encoding = np.dot(d_encoding_hidden, self.weights['encoding_hidden'].T) * self.sigmoid_derivative(encoded)
            d_encoder_hidden = np.dot(d_hidden_encoding, self.weights['hidden_encoding'].T) * self.sigmoid_derivative(hidden)
            
            self.weights['hidden_decoder'] += self.learning_rate * np.dot(hidden_decoded.T, d_hidden_decoder)
            self.biases['hidden_decoder'] += self.learning_rate * np.sum(d_hidden_decoder, axis=0, keepdims=True)
            self.weights['encoding_hidden'] += self.learning_rate * np.dot(encoded.T, d_encoding_hidden)
            self.biases['encoding_hidden'] += self.learning_rate * np.sum(d_encoding_hidden, axis=0, keepdims=True)
            self.weights['hidden_encoding'] += self.learning_rate * np.dot(hidden.T, d_hidden_encoding)
            self.biases['hidden_encoding'] += self.learning_rate * np.sum(d_hidden_encoding, axis=0, keepdims=True)
            self.weights['encoder_hidden'] += self.learning_rate * np.dot(X.T, d_encoder_hidden)
            self.biases['encoder_hidden'] += self.learning_rate * np.sum(d_encoder_hidden, axis=0, keepdims=True)
            
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')

    def encode(self, X):
        hidden = self.sigmoid(np.dot(X, self.weights['encoder_hidden']) + self.biases['encoder_hidden'])
        return self.sigmoid(np.dot(hidden, self.weights['hidden_encoding']) + self.biases['hidden_encoding'])

    def decode(self, encoded):
        hidden_decoded = self.sigmoid(np.dot(encoded, self.weights['encoding_hidden']) + self.biases['encoding_hidden'])
        return self.sigmoid(np.dot(hidden_decoded, self.weights['hidden_decoder']) + self.biases['hidden_decoder'])

    def reconstruct(self, X):
        encoded = self.encode(X)
        decoded = self.decode(encoded)
        return decoded
    
class MLP:
    def __init__(self, n_features, n_hidden, n_classes, learning_rate=0.01, epochs=1000):
        self.n_features = n_features
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights_input_hidden = np.random.randn(n_features, n_hidden)
        self.weights_hidden_output = np.random.randn(n_hidden, n_classes)
        self.bias_hidden = np.zeros((1, n_hidden))
        self.bias_output = np.zeros((1, n_classes))
        self.losses = []
        self.name = 'MLP'

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def _cross_entropy_loss(self, y, y_pred):
        epsilon = 1e-15
        return -np.mean(y * np.log(y_pred + epsilon) + (1-y) * np.log(1-y_pred + epsilon))

    def fit(self, X, y):
        for epoch in range(self.epochs):
            # 前向传播
            hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
            hidden_output = self.sigmoid(hidden_input)
            final_input = np.dot(hidden_output, self.weights_hidden_output) + self.bias_output
            final_output = self.softmax(final_input)

            # 计算损失
            # loss = -np.mean(y * np.log(final_output + 1e-15))
            loss = self._cross_entropy_loss(y, final_output)
            self.losses.append(loss)

            # 反向传播
            error_output = final_output - y
            error_hidden = np.dot(error_output, self.weights_hidden_output.T) * self.sigmoid_derivative(hidden_output)

            # 更新权重和偏置
            self.weights_hidden_output -= self.learning_rate * np.dot(hidden_output.T, error_output)
            self.bias_output -= self.learning_rate * np.sum(error_output, axis=0, keepdims=True)
            self.weights_input_hidden -= self.learning_rate * np.dot(X.T, error_hidden)
            self.bias_hidden -= self.learning_rate * np.sum(error_hidden, axis=0, keepdims=True)

            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')

    def predict(self, X):
        hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        hidden_output = self.sigmoid(hidden_input)
        final_input = np.dot(hidden_output, self.weights_hidden_output) + self.bias_output
        final_output = self.softmax(final_input)
        return np.argmax(final_output, axis=1) + 1
    
    def predict_binary(self, X):
        hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        hidden_output = self.sigmoid(hidden_input)
        final_input = np.dot(hidden_output, self.weights_hidden_output) + self.bias_output
        final_output = self.softmax(final_input)
        predictions = np.argmax(final_output, axis=1)
        predictions = np.where(predictions == 0, -1, 1)
        return predictions

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None
        self.losses = []
        self.name = 'SVM'

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)
        self.w = np.zeros(n_features)
        self.b = 0

        for epoch in range(self.n_iters):
            loss = 0
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                if condition:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w)
                else:
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.learning_rate * y_[idx]
                    loss += 1 - y_[idx] * (np.dot(x_i, self.w) - self.b)
                if epoch % 100 == 0:
                    print(f'Epoch {epoch}, Loss: {loss}')
            self.losses.append(loss / n_samples)


    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)
    
class SVM_Gaussian:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000, gamma=0.1):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.gamma = gamma
        self.alpha = None
        self.b = None
        self.X = None
        self.y = None
        self.losses = []
        self.name = 'SVM_Gaussian'

    def rbf_kernel(self, X1, X2):
        # return np.exp(-self.gamma * np.linalg.norm(X1[:, np.newaxis] - X2, axis=2) ** 2)
        return np.exp(-np.linalg.norm(X1[:, np.newaxis] - X2, axis=2) ** 2 / (2 * self.gamma ** 2))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.alpha = np.zeros(n_samples)
        self.b = 0
        self.X = X
        self.y = np.where(y <= 0, -1, 1)

        for _ in range(self.n_iters):
            loss = 0
            for i in range(n_samples):
                if self.y[i] * (np.sum(self.alpha * self.y * self.rbf_kernel(self.X, self.X[i:i+1])) + self.b) < 1:
                    self.alpha[i] += self.learning_rate * (1 - self.y[i] * (np.sum(self.alpha * self.y * self.rbf_kernel(self.X, self.X[i:i+1])) + self.b))
                    self.b += self.learning_rate * self.y[i]
                    loss += 1 - self.y[i] * (np.sum(self.alpha * self.y * self.rbf_kernel(self.X, self.X[i:i+1])) + self.b)
                # if epoch % 100 == 0:
                #     print(f'Epoch {epoch}, Loss: {loss}')
            self.losses.append(loss / n_samples)

    def predict(self, X):
        y_pred = np.sum(self.alpha[:, np.newaxis] * self.y[:, np.newaxis] * self.rbf_kernel(self.X, X), axis=0) + self.b
        return np.sign(y_pred)
    
class AdaBoost:
    def __init__(self, n_estimators=50, learning_rate=1.0):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.alphas = []
        self.models = []
        self.losses = []
        self.name = 'AdaBoost'

    def fit(self, X, y):
        n_samples, n_features = X.shape
        w = np.full(n_samples, (1 / n_samples))

        for _ in range(self.n_estimators):
            model = SVM(learning_rate=0.001, lambda_param=0.01, n_iters=100)
            model.fit(X, y * w)
            y_pred = model.predict(X)

            error = np.sum(w * (y != y_pred)) / np.sum(w)
            alpha = self.learning_rate * np.log((1 - error) / (error + 1e-10))

            w *= np.exp(alpha * (y != y_pred))
            w /= np.sum(w)

            # if epoch % 100 == 0:
            #         print(f'Epoch {epoch}, Loss: {error}')

            self.alphas.append(alpha)
            self.models.append(model)
            self.losses.append(error)

    def predict(self, X):
        model_preds = np.array([alpha * model.predict(X) for alpha, model in zip(self.alphas, self.models)])
        y_pred = np.sign(np.sum(model_preds, axis=0))
        return y_pred