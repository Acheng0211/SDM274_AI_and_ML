import numpy as np
import wandb

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

    def _gradient(self, inputs, outputs, groundtruths):
        batch_size = inputs.shape[0]
        grads = np.zeros_like(inputs)
    
        for i in range(batch_size):
            input = inputs[i]
            output = outputs[i]
            groundtruth = groundtruths[i]
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
            y_pred = self._predict(X)
            loss = self._loss(y, y_pred)
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
            if self.gd == 'SGD':
                i = np.random.randint(0, len(X))
                grad = self._gradient(np.expand_dims(X[i], axis=0), np.expand_dims(y_pred[i], axis=0), np.expand_dims(y[i], axis=0))
            else: 
                grad = self._gradient(X, y_pred, y)
            self.W -= self.lr * grad        
            if break_out:
                break_out = False
                break

    def _predict(self, X):
        return np.sign(X @ self.W)

    def evaluate(self, X_test, y_test):
        TP,FP,FN,TN = 0,0,0,0
        X_test = self._preprocess_data(X_test)
        for i in range(len(X_test)):
            y = np.sign(self._predict(X_test[i]))
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

class LogisticRegression:
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
        self.loss = []

    def _linear_tf(self, X):
        return  X @ self.W
    
    def _sigmoid(self, z):
        return 1/(1.+np.exp(-z))
    
    def _predict_probality(self, X):
        z = self._linear_tf(X)
        return self._sigmoid(z)
    
    def _loss(self, y, y_pred):
        epsilon = 1e-5
        loss = -np.mean(y * np.log(y_pred + epsilon) + (1 - y) * np.log(1-y_pred+epl))

    def _gradient(self, X, y_pred, y):
        return -(y - y_pred) @ X / y.size

    def _preprocess_data(self,X):
        m, n = X.shape
        X_ = np.empty([m, n+1])
        X_[:, 0] = 1
        X_[:, 1:] = X
        return X_
    
    # def _loss(self, y, y_pred):
    #     loss = y_pred * y
    #     loss_all = -loss[loss < 0]
    #     return np.sum(loss_all)

    # def _gradient(self, inputs, outputs, groundtruths):
    #     batch_size = inputs.shape[0]
    #     grads = np.zeros_like(inputs)
    
    #     for i in range(batch_size):
    #         input = inputs[i]
    #         output = outputs[i]
    #         groundtruth = groundtruths[i]
    #         grad = - groundtruth * input.reshape(-1, 1) if output * groundtruth < 0 else np.zeros_like(input)
    #         grads[i] = grad.reshape(-1)
    
    #     # 求平均梯度
    #     avg_grad = np.mean(grads, axis=0).reshape(-1, 1)
    #     return avg_grad
    
    def fit(self, X, y):
        if self.tol is not None:
            loss_old = np.inf

        X = self._preprocess_data(X)
        for epoch in range(self.epoch):
            y_pred = self._predict_probalitywwqww(X)
            loss = self._loss(y, y_pred)
            self.loss.append(loss)
            if self.wandb:
                wandb.log({"loss": loss})
            if self.tol is not None:
                if np.abs(loss_old - loss) < self.tol:
                    break
                loss_old = loss
            
            if self.gd == 'SGD':
                i = np.random.randint(0, len(X))
                grad = self._gradient(np.expand_dims(X[i], axis=0), np.expand_dims(y_pred[i], axis=0), np.expand_dims(y[i], axis=0))
            else: 
                grad = self._gradient(X, y_pred, y)
            self.W -= self.lr * grad        


    def _predict(self, X): 
        # return np.sign(X @ self.W)
        X = self._preprocess_data(X)
        y_pred = self._predict_probality(X)
        return np.where(y_pred>=0.5,1,0)

    def evaluate(self, X_test, y_test):
        TP,FP,FN,TN = 0,0,0,0
        X_test = self._preprocess_data(X_test)
        for i in range(len(X_test)):
            y = np.sign(self._predict(X_test[i]))
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