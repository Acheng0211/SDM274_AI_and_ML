import numpy as np
import pandas as pd
import utils
from model import pca, LinearAutoencoder, NonLinearAutoencoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


if __name__ == '__main__':

    file_path = './Final_Project/seeds_dataset.txt'
    X_train, X_test, y_train, y_test = utils.load_data(file_path)

    # K-Means++



    # Soft K-Means



    # PCA
    X_pca, X_reconstructed_pca = pca(X_train, X_test)
    plt.scatter(X_reconstructed_pca[:, 0], X_reconstructed_pca[:, 1], c=y_train)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA of Wine Dataset')
    plt.show()

    # Linear_Autoencoder
    input_dim = X_train.shape[1]
    encoding_dim = 2
    linear_autoencoder = LinearAutoencoder(input_dim, encoding_dim, learning_rate=0.01, epochs=80000)
    linear_autoencoder.fit(X_train)
    X_reconstructed_linear = linear_autoencoder.reconstruct(X_test)
    encoded_linear = linear_autoencoder.encode(X_test)
    plt.scatter(encoded_linear[:, 0], encoded_linear[:, 1], c=y_test)
    plt.xlabel('Encoded Feature 1')
    plt.ylabel('Encoded Feature 2')
    plt.title('Linear Autoencoder of Wine Dataset')
    plt.show()

    # Non-linear Autoencoder
    hidden_dim = 16
    nonlinear_autoencoder = NonLinearAutoencoder(input_dim, encoding_dim, hidden_dim, learning_rate=0.001, epochs=80000)
    nonlinear_autoencoder.fit(X_train)
    X_reconstructed_nonlinear = nonlinear_autoencoder.reconstruct(X_test)
    encoded_nonlinear = nonlinear_autoencoder.encode(X_test)
    plt.scatter(encoded_nonlinear[:, 0], encoded_nonlinear[:, 1], c=y_test)
    plt.xlabel('Encoded Feature 1')
    plt.ylabel('Encoded Feature 2')
    plt.title('Non-linear Autoencoder of Wine Dataset')
    plt.show()

    # 计算重构误差
    error_pca = utils.reconstruction_error(X_train, X_reconstructed_pca)
    error_linear = utils.reconstruction_error(X_test, X_reconstructed_linear)
    error_nonlinear = utils.reconstruction_error(X_test, X_reconstructed_nonlinear)

    print(f'Reconstruction Error (PCA): {error_pca}')
    print(f'Reconstruction Error (Linear Autoencoder): {error_linear}')
    print(f'Reconstruction Error (Non-linear Autoencoder): {error_nonlinear}')


    # Clustering



    # MLP for Multi-class Classification



    # SVM and SVM with Gaussian Kernel



    # AdaBoost Algorithm



    # Binary Classification
