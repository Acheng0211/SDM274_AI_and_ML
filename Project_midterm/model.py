import numpy as np 
import wandb
import matplotlib as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

class LinearRegression:
    def __init__(self, n_feature = 1, lr=1e-5, epoch=5000, batch_size=10, tol = 1e-5, gd = None):
        self.lr = lr  # 学习率
        self.epoch = epoch    # 迭代次数
        self.batch_size = batch_size        # 批量大小
        self.tol = tol          # 收敛阈值
        self.W = (np.random.randn(n_feature + 1) * 0.05).reshape(-1,1)    # 权重
        self.loss = []
        self.gd = gd
        self.norm_params = None  # 用于保存归一化参数

    def fit(self, X, y):
        # # 无归一化
        X_norm = np.c_[X, np.ones(X.shape[0])]  # 添加一列全为1的列作为偏置   

        # # 对x进行归一化
        # self.norm_params = (X.min(axis=0), X.max(axis=0))  # 保存最小值和最大值
        # X_norm = self.min_max_normalization(X)  # 对X进行归一化
        # # X_norm = self.mean_normalization(X)
        # X_norm = np.c_[X_norm, np.ones(X_norm.shape[0])]  # 添加一列全为1的列作为偏置
               
        for i in range(self.epoch):
            if self.gd == 'SGD':
                self._sgd_update(X_norm, y)
            elif self.gd == 'BGD':
                self._bgd_update(X_norm, y)
            elif self.gd == 'MBGD':
                self._mbgd_update(X_norm, y)
            if i % 100 == 0:
                loss = self._mse_loss(y, np.dot(X_norm,self.W))
                self.loss.append(loss)
                print(f"for {i} iteration, the loss is {loss}")
        return self.W

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def _predict(self, X):
        y_pred = self._sigmoid(X @ self.W)
        return y_pred

    def _mse_loss(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)  # 计算均方误差损失
        
    def _gradient(self, X, y, y_pred):  
        grad = 1 / X.shape[0] * np.dot(X.T, (y_pred - y))
        # grad = grad.reshape(self.W.shape)
        return grad

    def _sgd_update(self, X, y): #epoch=2750
        pred = X @ self.W 
        i = np.random.randint(0, len(X))
        grad = self._gradient(X[i,:][np.newaxis, :], y[i], pred[i])
        self.W -= self.lr * grad  # 随机梯度下降更新

    def _bgd_update(self, X, y): #lr=1e-6
        pred = X @ self.W 
        grad = self._gradient(X, y, pred)
        self.W -= self.lr * grad  # 批量梯度下降更新
        
    def _mbgd_update(self, X, y): #epoch=10000
        pred = self._predict(X)
        indices = np.random.choice(y.shape[0], self.batch_size, replace=False)
        grad = self._gradient(X[indices], y[indices], pred[indices])
        grad = np.asarray(grad, dtype=np.float64)
        self.W -= self.lr * grad/self.batch_size

    def predict(self, X):
        # 使用保存的归一化参数对新的输入数据X进行归一化
        X_norm = self.min_max_normalization(X, None)
        return self._predict(X_norm)

    def _evaluate(self, X_test, y_test):
        scores = {'accuracy': [], 'recall': [], 'precision': [], 'f1': []}
        X_test = np.c_[X_test, np.ones(X_test.shape[0])]
        predictions = self._predict(X_test)
        threshold = 0.5
        predictions = (predictions >= threshold).astype(np.int64)
        np.savetxt("compare.csv", np.concatenate((y_test, predictions), axis=1), delimiter=", ")
        scores['accuracy'].append(accuracy_score(y_test, predictions))
        scores['recall'].append(recall_score(y_test, predictions, average='weighted', zero_division=0))
        scores['precision'].append(precision_score(y_test, predictions, average='weighted', zero_division=0))
        scores['f1'].append(f1_score(y_test, predictions, average='weighted', zero_division=0))
        return {metric: np.mean(values) for metric, values in scores.items()}
    
    def min_max_normalization(self, X, params=None):
        if params is None:
            params = (X.min(axis=0), X.max(axis=0))
        X_min, X_max = params
        return (X - X_min) / (X_max - X_min)

    def mean_normalization(self, X, params=None):
        if params is None:
            X_mean, X_std = X.mean(axis=0), np.std(X, axis=0)
        else:
            X_mean, X_std = params
        return (X - X_mean) / X_std
    
class Perceptron:
    def __init__(self, n_feature = 1, epoch = 2000, lr = 0.01, tol = None, wandb = False, gd = None):
        self.n_feature = n_feature
        self.epoch = epoch
        self.lr = lr
        self.tol = tol
        self.wandb = wandb
        self.gd = gd
        self.W = (np.random.rand(n_feature + 1) * 0.5).reshape(-1,1)
        self.best_loss = np.inf
        self.patience = 100
        self.loss =[]

    def _preprocess_data(self,X):
        m, n = X.shape
        X_ = np.empty([m, n+1])
        X_[:, 0] = 1
        X_[:, 1:] = X
        return X_
    
    def _loss(self, y, y_pred):
        loss = y_pred * y
        loss_all = -loss[loss < 0]
        return np.sum(loss_all)

    def _gradient(self, X, y, y_pred):
        batch_size = X.shape[0]
        grads = np.zeros_like(X)
    
        for i in range(batch_size):
            input = X[i]
            output = y[i]
            groundtruth = y_pred[i]
            grad = - groundtruth * input.reshape(-1, 1) if output * groundtruth < 0 else np.zeros_like(input)
            grads[i] = grad.reshape(-1)
    
        # 求平均梯度
        avg_grad = np.mean(grads, axis=0).reshape(-1, 1)
        return avg_grad
    
    def fit(self, X, y):
        break_out = False
        epoch_no_improve = 0
        X = self._preprocess_data(X)
        
        for epoch in range(self.epoch):
            if self.gd == 'SGD':
                i = np.random.randint(0, len(X))
                y_pred = self._predict(X[i])
                grad = self._gradient(np.expand_dims(X[i], axis=0), np.expand_dims(y_pred[i], axis=0), np.expand_dims(y[i], axis=0))
                loss = self._loss(y[i], y_pred)
            elif self.gd == 'MBGD':
                batch_size = 10
                idx = np.random.choice(y.shape[0], batch_size, replace=False)
                y_pred = self._predict(X[idx])
                grad = self._gradient(X[idx], y[idx], y_pred) 
                loss = self._loss(y[idx], y_pred)
                self.loss.append(loss)
            self.W -= self.lr * grad
            if epoch % 100 == 0:            
                print(f'Epoch {epoch}, Loss: {loss}')
            if self.wandb:
                wandb.log({"loss": loss})
            if loss < self.best_loss - self.tol:
                self.best_loss = loss
                epoch_no_improve = 0
            elif np.abs(loss - self.best_loss) < self.tol:
                epoch_no_improve += 1
                if epoch_no_improve >= self.patience:
                    print(f"Early stopping triggered at {epoch} due to the no improvement in loss")
                    break_out = True
                    break
                else:
                    epoch_no_improve = 0        
            if break_out:
                break_out = False
                break

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def _predict(self, X): 
        y_pred = self._sigmoid(X @ self.W)
        y_pred_binary = np.where(y_pred>=0.95,1,-1)
        return y_pred_binary

    def _evaluate(self, X_test, y_test):
        scores = {'accuracy': [], 'recall': [], 'precision': [], 'f1': []}
        X_test = self._preprocess_data(X_test)
        predictions = self._predict(X_test)
        threshold = 0
        predictions = (predictions >= threshold).astype(np.int64)
        # predictions = np.round(self._predict(X_test))
        np.savetxt("compare1.csv", np.concatenate((y_test, predictions), axis=1), delimiter=", ")
        scores['accuracy'].append(accuracy_score(y_test, predictions))
        scores['recall'].append(recall_score(y_test, predictions, average='weighted', zero_division=0))
        scores['precision'].append(precision_score(y_test, predictions, average='weighted', zero_division=0))
        scores['f1'].append(f1_score(y_test, predictions, average='weighted', zero_division=0))
        return {metric: np.mean(values) for metric, values in scores.items()}

    def evaluate(self, X_test, y_test):
        TP,FP,FN,TN = 0,0,0,0
        X_test = self._preprocess_data(X_test)
        for i in range(len(X_test)):
            y = self._predict(X_test[i])
            if y == 1 and y_test[i] == 1:
                TP += 1
            elif y == 1 and y_test[i] == -1:
                FP += 1
            elif y == -1 and y_test[i] == 1:
                FN += 1
            elif y == -1 and y_test[i] == -1:
                TN += 1
        # print(TP, FP, FN, TN)
        accuracy = (TP + TN) / (TP + FP + FN + TN)
        recall = TP / (TP + FN)
        precision = TP / (TP + FP)
        F1 = 2 * precision * recall / (precision + recall)
        print(f"evaluation results: accuracy: {accuracy}, recall: {recall}, precision: {precision}, F1: {F1}")

    def plot_loss(self, loss, name):
        plt.plot(loss)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{name}_{self.gd}_loss')
        # plt.savefig(os.path.join('./output', f'{name}_{self.gd}_loss.png'))
        plt.show()

class LogisticRegression:
    def __init__(self, n_feature = 1, epoch = 2000, lr = 0.01, tol = None, wandb = False, gd = None):
        self.n_feature = n_feature
        self.epoch = epoch
        self.lr = lr
        self.tol = tol
        self.wandb = wandb
        self.gd = gd
        self.W = (np.random.rand(n_feature + 1) * 0.05).reshape(-1,1)
        self.best_loss = np.inf
        self.patience = 100
        self.loss = []

    def _linear_tf(self, X):
        return X @ self.W
    
    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def _predict_probablity(self, X): 
        z = self._linear_tf(X)
        prob = self._sigmoid(z)
        # data = np.concatenate((z, prob), axis=1)
        # np.savetxt("compare.csv", data, delimiter=", ")
        return prob
    
    def _loss(self, y, y_pred):
        epsilon = 1e-5
        loss = -np.mean(y * np.log(y_pred + epsilon) + (1-y) * np.log(1-y_pred + epsilon))
        return loss

    def _gradient(self, X, y, y_pred):
        return (-(y - y_pred) @ X / y.size).reshape(-1, 1)

    def _preprocess_data(self,X):
        m, n = X.shape
        X_ = np.empty([m, n+1])
        X_[:, 0] = 1
        X_[:, 1:] = X
        return X_
    
    def fit(self, X, y): 
        break_out = False
        epoch_no_improve = 0

        X = self._preprocess_data(X)
        for epoch in range(self.epoch):
            if self.gd == 'SGD':
                i = np.random.randint(0, len(X))
                y_pred = self._predict_probablity(X[i])
                grad = self._gradient(np.expand_dims(X[i], axis=0), np.expand_dims(y[i], axis=0), np.expand_dims(y_pred[i], axis=0))
                loss = self._loss(y[i], y_pred)
            elif self.gd == 'MBGD':
                batch_size = 1
                idx = np.random.choice(y.shape[0], batch_size, replace=False)
                y_pred = self._predict_probablity(X[idx])
                grad = self._gradient(X[idx], y[idx], y_pred) 
                loss = self._loss(y[idx], y_pred)
            self.W -= self.lr * grad 
            self.loss.append(loss)
            if epoch % 100 == 0:            
                print(f'Epoch {epoch}, Loss: {loss}')
            loss_error = np.abs(loss - self.best_loss)
            if self.wandb:
                wandb.log({"loss": loss})
            if loss < self.best_loss - self.tol:
                self.best_loss = loss
                epoch_no_improve = 0
            elif loss_error < self.tol:
                epoch_no_improve += 1
                if epoch_no_improve >= self.patience:
                    # print(f'loss error is {loss_error}, smaller than tolerance, quit at epoch {epoch}')
                    print(f"Early stopping triggered at {epoch} due to the no improvement in loss")
                    break_out = True
                    break
                else:
                    epoch_no_improve = 0

            if break_out:
                break_out = False
                break

    def _predict(self, X): 
        X = self._preprocess_data(X)
        y_pred = self._predict_probablity(X)
        return np.where(y_pred>=0.75,1,0)
    
    def _evaluate(self, X_test, y_test):
        scores = {'accuracy': [], 'recall': [], 'precision': [], 'f1': []}
        # X_test = self._preprocess_data(X_test)
        predictions = self._predict(X_test)
        np.savetxt("compare2.csv", np.concatenate((y_test, predictions), axis=1), delimiter=", ")
        scores['accuracy'].append(accuracy_score(y_test, predictions))
        scores['recall'].append(recall_score(y_test, predictions, average='weighted', zero_division=0))
        scores['precision'].append(precision_score(y_test, predictions, average='weighted', zero_division=0))
        scores['f1'].append(f1_score(y_test, predictions, average='weighted', zero_division=0))
        return {metric: np.mean(values) for metric, values in scores.items()}

    def evaluate(self, X_test, y_test):
        TP,FP,FN,TN = 0,0,0,0
        X_test = self._predict(X_test)
        for i in range(len(X_test)):
            y = X_test[i]
            if y == 1 and y_test[i] == 1:
                TP += 1
            elif y == 1 and y_test[i] == -1:
                FP += 1
            elif y == 0 and y_test[i] == 1:
                FN += 1
            elif y == 0 and y_test[i] == -1:
                TN += 1
        # print(TP, FP, FN, TN)
        accuracy = (TP + TN) / (TP + FP + FN + TN)
        recall = TP / (TP + FN)
        precision = TP / (TP + FP)
        F1 = 2 * precision * recall / (precision + recall)
        print(f"evaluation results:\n    accuracy: {accuracy}, \n    recall: {recall}, \n    precision: {precision}, \n    F1: {F1}")
    
    def plot_loss(self, loss, name):
        plt.plot(loss)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{name}_{self.gd}_loss')
        # plt.savefig(os.path.join('./output', f'{name}_{self.gd}_loss.png'))
        plt.show()

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
            self.biases.append(np.zeros((1, self.layers[i + 1])))

    def sigmoid(self, x):
        x = np.asarray(x, dtype=np.float64)  # 确保输入是数值类型
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward_propagation(self, X):
        self.a = [X]
        for i in range(len(self.weights) - 1):
            net = np.dot(self.a[-1], self.weights[i]) + self.biases[i]
            self.a.append(self.sigmoid(net))
        # 最后一层不使用激活函数
        net = np.dot(self.a[-1], self.weights[-1]) + self.biases[-1]
        self.a.append(net)
        return self.a[-1]

    def backward_propagation(self, X, y, learning_rate):
        self.a[-1] = np.asarray(self.a[-1], dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        m = y.shape[0]
        self.deltas = [self.a[-1] - y]
        for i in range(len(self.a) - 2, 0, -1):
            # fixed by y.reshape(-1,1)
            delta = np.dot(self.deltas[-1], self.weights[i].T) * self.sigmoid_derivative(self.a[i])
            self.deltas.append(delta)
        self.deltas.reverse()
        for i in range(len(self.weights)):
            self.weights[i] -= learning_rate * np.dot(self.a[i].T, self.deltas[i]) / m
            self.biases[i] -= learning_rate * np.sum(self.deltas[i], axis=0, keepdims=True) / m

    def train(self, X, y, epochs, learning_rate, batch_size=None, gd='MBGD'):
        self.gd = gd
        if gd not in ['SGD', 'MBGD']:
            raise ValueError("gd should be either 'SGD' or 'MBGD'")
        
        if gd == 'SGD':
            batch_size = 1
        elif gd == 'MBGD' and batch_size is None:
            batch_size = X.shape[0]
        
        for epoch in range(epochs):
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            for start in range(0, X.shape[0], batch_size):
                end = start + batch_size
                X_batch, y_batch = X[indices[start:end]], y[indices[start:end]]
                X_batch = np.asarray(X_batch, dtype=np.float64)
                y_batch = np.asarray(y_batch, dtype=np.float64)
                self.forward_propagation(X_batch)
                self.backward_propagation(X_batch, y_batch, learning_rate)
            loss = np.mean((self.forward_propagation(X) - y) ** 2)
            self.loss.append(loss)
            if epoch % 100 == 0:            
                print(f'Epoch {epoch}, Loss: {loss}')

    def predict(self, X):
        return self.forward_propagation(X)
    
    def evaluate(self, X_test, y_test):
        TP,FP,FN,TN = 0,0,0,0
        y_pred = self.predict(X_test)
        y_pred_binary = np.where(y_pred>=0.05,1,0)
        for i in range(len(y_pred)):
            y = y_pred_binary[i]
            if y == 1 and y_test[i] == 1:
                TP += 1
            elif y == 1 and y_test[i] == 0:
                FP += 1
            elif y == 0 and y_test[i] == 1:
                FN += 1
            elif y == 0 and y_test[i] == 0:
                TN += 1
        # print(TP, FP, FN, TN)
        accuracy = (TP + TN) / (TP + FP + FN + TN)
        recall = TP / (TP + FN)
        precision = TP / (TP + FP)
        F1 = 2 * precision * recall / (precision + recall)
        print(f"evaluation results:\n    accuracy: {accuracy}, \n    recall: {recall}, \n    precision: {precision}, \n    F1: {F1}")
        scores = {'accuracy': [accuracy], 'recall': [recall], 'precision': [precision], 'f1': [F1]}
        return {metric: np.mean(values) for metric, values in scores.items()}