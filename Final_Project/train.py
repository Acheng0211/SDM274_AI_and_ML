import os
import numpy as np
import pandas as pd
import utils
from model import PCA, NonLinearAutoencoder, MLP, SVM, SVM_Gaussian, AdaBoost
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# K-Means++
def initialize_centroids(X, K):
    centroids = [X[np.random.randint(X.shape[0])]]
    for _ in range(1, K):
        distances = np.array([np.min([np.linalg.norm(x - centroid) for centroid in centroids]) for x in X])
        probabilities = distances / np.sum(distances)
        cumulative_probabilities = np.cumsum(probabilities)
        r = np.random.rand()
        for i, p in enumerate(cumulative_probabilities):
            if r < p:
                centroids.append(X[i])
                break
    return np.array(centroids)

def kmeans_plus_plus(X, K, max_iters=100):
    centroids = initialize_centroids(X, K)
    for _ in range(max_iters):
        clusters = [[] for _ in range(K)]
        for x in X:
            distances = [np.linalg.norm(x - centroid) for centroid in centroids]
            cluster = np.argmin(distances)
            clusters[cluster].append(x)
        new_centroids = np.array([np.mean(cluster, axis=0) for cluster in clusters])
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return centroids, clusters

# Soft K-Means
def soft_kmeans(X, K, beta=1.0, max_iters=100):
    centroids = X[np.random.choice(X.shape[0], K, replace=False)]
    for _ in range(max_iters):
        distances = np.array([[np.linalg.norm(x - centroid) for centroid in centroids] for x in X])
        responsibilities = np.exp(-beta * distances)
        responsibilities /= np.sum(responsibilities, axis=1, keepdims=True)
        new_centroids = np.dot(responsibilities.T, X) / np.sum(responsibilities, axis=0)[:, np.newaxis]
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return centroids, responsibilities

def plot_clusters(X, clusters, centroids, title):
    for i, cluster in enumerate(clusters):
        cluster = np.array(cluster)
        plt.scatter(cluster[:, 0], cluster[:, 1], label=f'Cluster {i+1}')
    plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='red', label='Centroids')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.legend()
    plt.savefig(os.path.join('./Final_Project/output', f'{title}.png'))
    plt.show()

def plot_soft_clusters(X, responsibilities, centroids, title):
    for i in range(centroids.shape[0]):
        cluster = X[np.argmax(responsibilities, axis=1) == i]
        plt.scatter(cluster[:, 0], cluster[:, 1], label=f'Cluster {i+1}')
    plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='red', label='Centroids')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(title)
    plt.legend()
    plt.savefig(os.path.join('./Final_Project/output', f'{title}.png'))
    plt.show()

def calculate_clustering_accuracy(y_true, y_pred):
    # confusion matrix
    D = max(y_pred.max(), y_true.max()) + 1
    confusion_matrix = np.zeros((D, D), dtype=np.int32)
    for i in range(len(y_true)):
        confusion_matrix[y_pred[i], y_true[i]] += 1

    # hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(confusion_matrix.max() - confusion_matrix)
    accuracy = confusion_matrix[row_ind, col_ind].sum() / len(y_true)
    return accuracy

# todo: nonlinear autoencoder optimization and binary classification check
if __name__ == '__main__':

    file_path = './Final_Project/seeds_dataset.txt'
    X_train, X_test, y_train, y_test = utils.load_data(file_path)

# K-Means++ clustering
    K = 3
    centroids, clusters = kmeans_plus_plus(X_train, K)

    # visualize k-means clustering
    for i, cluster in enumerate(clusters):
        cluster = np.array(cluster)
        plt.scatter(cluster[:, 0], cluster[:, 1], label=f'Cluster {i+1}')
    plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='red', label='Centroids')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('K-Means++ Clustering')
    plt.legend()
    plt.savefig(os.path.join('./Final_Project/output', f'K-Means++_Clustering.png'))
    plt.show()

    y_pred_kmeans = np.concatenate([[i] * len(cluster) for i, cluster in enumerate(clusters)])
    accuracy_kmeans = calculate_clustering_accuracy(y_train, y_pred_kmeans)
    print(f'K-Means Clustering Accuracy: {accuracy_kmeans}')

# Soft K-Means clustering
    K = 3
    centroids, responsibilities = soft_kmeans(X_train, K)

    # visualize soft k-means clustering
    for i in range(K):
        cluster = X_train[np.argmax(responsibilities, axis=1) == i]
        plt.scatter(cluster[:, 0], cluster[:, 1], label=f'Cluster {i+1}')
    plt.scatter(centroids[:, 0], centroids[:, 1], s=200, c='red', label='Centroids')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Soft K-Means Clustering')
    plt.legend()
    plt.savefig(os.path.join('./Final_Project/output', f'Soft_K-Means_Clustering.png'))
    plt.show()

    y_pred_soft_kmeans = np.argmax(responsibilities, axis=1)
    accuracy_soft_kmeans = calculate_clustering_accuracy(y_train, y_pred_soft_kmeans)
    print(f'Soft K-Means Clustering Accuracy: {accuracy_soft_kmeans}')

# PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_train)

    # visualize PCA
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA of Seeds Dataset')
    plt.savefig(os.path.join('./Final_Project/output', f'PCA.png'))
    plt.show()

# Non-linear Autoencoder
    hidden_dim = 32
    input_dim = X_train.shape[1]
    encoding_dim = 2
    nonlinear_autoencoder = NonLinearAutoencoder(input_dim, encoding_dim, hidden_dim, learning_rate=0.01, epochs=8000)
    nonlinear_autoencoder.fit(X_train)
    X_encoded_nonlinear = nonlinear_autoencoder.encode(X_train)

    # visualize non-linear autoencoder
    plt.figure()
    plt.scatter(X_encoded_nonlinear[:, 0], X_encoded_nonlinear[:, 1], c=y_train)
    plt.xlabel('Encoded Feature 1')
    plt.ylabel('Encoded Feature 2')
    plt.title('Non-linear Autoencoder of Seeds Dataset')
    plt.savefig(os.path.join('./Final_Project/output', f'Non-linear_Autoencoder.png'))
    plt.show()

    utils.plot_loss(nonlinear_autoencoder.losses, nonlinear_autoencoder.name)

# Clustering with Reduced Dimensions
    # Apply K-Means++ and Soft K-Means on PCA reduced data
    K = 3
    centroids_pca_kmeans, clusters_pca_kmeans = kmeans_plus_plus(X_pca, K)
    centroids_pca_soft, responsibilities_pca_soft = soft_kmeans(X_pca, K)

    # Apply K-Means++ and Soft K-Means on Non-linear Autoencoder reduced data
    centroids_nonlinear_kmeans, clusters_nonlinear_kmeans = kmeans_plus_plus(X_encoded_nonlinear, K)
    centroids_nonlinear_soft, responsibilities_nonlinear_soft = soft_kmeans(X_encoded_nonlinear, K)

    # visualize K-Means++ clustering on reduced data
    plot_clusters(X_pca, clusters_pca_kmeans, centroids_pca_kmeans, 'K-Means++ on PCA Reduced Data')
    plot_clusters(X_encoded_nonlinear, clusters_nonlinear_kmeans, centroids_nonlinear_kmeans, 'K-Means++ on Non-linear Autoencoder Reduced Data')

    # visualize Soft K-Means clustering on reduced data
    plot_soft_clusters(X_pca, responsibilities_pca_soft, centroids_pca_soft, 'Soft K-Means on PCA Reduced Data')
    plot_soft_clusters(X_encoded_nonlinear, responsibilities_nonlinear_soft, centroids_nonlinear_soft, 'Soft K-Means on Non-linear Autoencoder Reduced Data')

    # calculate clustering accuracy on PCA reduced data
    y_pred_clusters_pca_kmeans = np.concatenate([[i] * len(cluster) for i, cluster in enumerate(clusters_pca_kmeans)])
    accuracy_clusters_pca_kmeans = calculate_clustering_accuracy(y_train, y_pred_clusters_pca_kmeans)
    print(f'K-Means Clustering with PCA Reduced Dimensions Accuracy: {accuracy_clusters_pca_kmeans}')
    y_pred_pca_soft_kmeans = np.argmax(responsibilities_pca_soft, axis=1)
    accuracy_pca_soft_kmeans = calculate_clustering_accuracy(y_train, y_pred_pca_soft_kmeans)
    print(f'Soft K-Means Clustering with PCA Reduced Dimensions Accuracy: {accuracy_pca_soft_kmeans}')
    y_pred_clusters_nonlinear_kmeans = np.concatenate([[i] * len(cluster) for i, cluster in enumerate(clusters_nonlinear_kmeans)])
    accuracy_clusters_nonlinear_kmeans = calculate_clustering_accuracy(y_train, y_pred_clusters_nonlinear_kmeans)
    print(f'K-Means Clustering with Nonlinear Autoencoder Reduced Dimensions Accuracy: {accuracy_clusters_nonlinear_kmeans}')
    y_pred_nonlinear_soft_kmeans = np.argmax(responsibilities_nonlinear_soft, axis=1)
    accuracy_nonlinear_soft_kmeans = calculate_clustering_accuracy(y_train, y_pred_nonlinear_soft_kmeans)
    print(f'Soft K-Means Clustering with Nonlinear Autoencoder Reduced Dimensions Accuracy: {accuracy_nonlinear_soft_kmeans}')


# MLP for Multi-class Classification
    # one-hot coding
    encoder = OneHotEncoder(sparse_output=False)
    y_train_onehot = encoder.fit_transform(y_train.reshape(-1, 1))
    y_test_onehot = encoder.transform(y_test.reshape(-1, 1))

    # train MLP
    n_features = X_train.shape[1]
    n_hidden = 16
    n_classes = len(np.unique(y_train))
    mlp = MLP(n_features, n_hidden, n_classes, learning_rate=0.01, epochs=10000)
    mlp.fit(X_train, y_train_onehot)

    # evaluate MLP
    y_pred = mlp.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print(f'MLP for Multi-class Classification Accuracy: {accuracy}')

    utils.plot_loss(mlp.losses, mlp.name)

# Binary Classification
    # remoce the data with label 2
    binary_mask = y_train != 2
    X_train_binary = X_train[binary_mask]
    y_train_binary = y_train[binary_mask]
    y_train_binary = np.where(y_train_binary == 1, 1, -1)

    binary_mask = y_test != 2
    X_test_binary = X_test[binary_mask]
    y_test_binary = y_test[binary_mask]
    y_test_binary = np.where(y_test_binary == 1, 1, -1)

    # train and evaluate MLP
    encoder1 = OneHotEncoder(sparse_output=False)
    y_train_binary_onehot = encoder1.fit_transform(y_train_binary.reshape(-1, 1))
    y_test_binary_onehot = encoder1.transform(y_test_binary.reshape(-1, 1))
    n_features = X_train_binary.shape[1]
    n_hidden = 16
    n_classes = len(np.unique(y_train_binary))
    mlp = MLP(n_features, n_hidden, n_classes, learning_rate=0.01, epochs=10000)
    mlp.fit(X_train_binary, y_train_binary_onehot)
    y_pred_mlp = mlp.predict_binary(X_test_binary)
    accuracy_mlp = np.mean(y_pred_mlp == y_test_binary)
    print(f'MLP Accuracy: {accuracy_mlp}')
    utils.plot_loss(mlp.losses, mlp.name, remarks='_binary_')

    # train and evaluate SVM
    svm = SVM(learning_rate=0.001, lambda_param=0.01, n_iters=1000)
    svm.fit(X_train_binary, y_train_binary)
    y_pred_svm = svm.predict(X_test_binary)
    accuracy_svm = np.mean(y_pred_svm == y_test_binary)
    print(f'SVM Accuracy: {accuracy_svm}')
    utils.plot_loss(svm.losses, svm.name, remarks='_binary_')

    # train and evaluate SVM with Gaussian Kernel
    svm_gaussian = SVM_Gaussian(learning_rate=0.001, lambda_param=0.01, n_iters=1000, gamma=0.1)
    svm_gaussian.fit(X_train_binary, y_train_binary)
    y_pred_svm_gaussian = svm_gaussian.predict(X_test_binary)
    accuracy_svm_gaussian = np.mean(y_pred_svm_gaussian == y_test_binary)
    print(f'SVM with Gaussian Kernel Accuracy: {accuracy_svm_gaussian}')
    # utils.plot_loss(svm_gaussian.losses, svm_gaussian.name, remarks='_binary_')

    # train and evaluate AdaBoost
    adaboost = AdaBoost(n_estimators=50, learning_rate=1.0)
    adaboost.fit(X_train_binary, y_train_binary)
    y_pred_adaboost = adaboost.predict(X_test_binary)
    accuracy_adaboost = np.mean(y_pred_adaboost == y_test_binary)
    print(f'AdaBoost Accuracy: {accuracy_adaboost}')
    # utils.plot_loss(adaboost.losses, adaboost.name, remarks='_binary_')

# print all the accuracies
    print(f'K-Means Clustering Accuracy: {accuracy_kmeans}')
    print(f'Soft K-Means Clustering Accuracy: {accuracy_soft_kmeans}')
    print(f'K-Means Clustering with PCA Reduced Dimensions Accuracy: {accuracy_clusters_pca_kmeans}')
    print(f'Soft K-Means Clustering with PCA Reduced Dimensions Accuracy: {accuracy_pca_soft_kmeans}')
    print(f'K-Means Clustering with Nonlinear Autoencoder Reduced Dimensions Accuracy: {accuracy_clusters_nonlinear_kmeans}')
    print(f'Soft K-Means Clustering with Nonlinear Autoencoder Reduced Dimensions Accuracy: {accuracy_nonlinear_soft_kmeans}')
    print(f'MLP for Multi-class Classification Accuracy: {accuracy}')
    print(f'MLP Accuracy: {accuracy_mlp}')
    print(f'SVM Accuracy: {accuracy_svm}')
    print(f'SVM with Gaussian Kernel Accuracy: {accuracy_svm_gaussian}')
    print(f'AdaBoost Accuracy: {accuracy_adaboost}')