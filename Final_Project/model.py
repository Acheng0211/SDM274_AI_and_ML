import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def pca(X_train, X_test, n_components=2):
    covariance_matrix = np.cov(X_train.T)
    
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]
    principal_components = eigenvectors[:, :n_components]
    
    X_pca = np.dot(X_train, principal_components)
    
    X_reconstructed = np.dot(X_pca, principal_components.T)
    
    return X_pca, X_reconstructed

class LinearAutoencoder:
    def __init__(self, input_dim, encoding_dim, learning_rate=0.01, epochs=1000):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = {
            'encoder': np.random.randn(input_dim, encoding_dim),
            'decoder': np.random.randn(encoding_dim, input_dim)
        }
        self.biases = {
            'encoder': np.zeros((1, encoding_dim)),
            'decoder': np.zeros((1, input_dim))
        }

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def fit(self, X):
        for epoch in range(self.epochs):
            encoded = self.sigmoid(np.dot(X, self.weights['encoder']) + self.biases['encoder'])
            decoded = self.sigmoid(np.dot(encoded, self.weights['decoder']) + self.biases['decoder'])
            
            loss = np.mean((X - decoded) ** 2)
            
            error = X - decoded
            d_decoder = error * self.sigmoid_derivative(decoded)
            d_encoder = np.dot(d_decoder, self.weights['decoder'].T) * self.sigmoid_derivative(encoded)
            
            self.weights['decoder'] += self.learning_rate * np.dot(encoded.T, d_decoder)
            self.biases['decoder'] += self.learning_rate * np.sum(d_decoder, axis=0, keepdims=True)
            self.weights['encoder'] += self.learning_rate * np.dot(X.T, d_encoder)
            self.biases['encoder'] += self.learning_rate * np.sum(d_encoder, axis=0, keepdims=True)
            
            if epoch % 100 == 0:
                print(f'Epoch {epoch}, Loss: {loss}')

    def encode(self, X):
        return self.sigmoid(np.dot(X, self.weights['encoder']) + self.biases['encoder'])

    def decode(self, encoded):
        return self.sigmoid(np.dot(encoded, self.weights['decoder']) + self.biases['decoder'])

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



