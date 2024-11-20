import hydra
import hydra.conf
import wandb
import numpy as np
import os
from omegaconf import DictConfig
import utils
from model import LinearRegression, Perceptron, LogisticRegression, MLP

@hydra.main(version_base="1.3", config_path="./conf", config_name="config_proj_midterm")
def main(cfg: DictConfig):
    # Preprocess dataset
    dataset_path = cfg.dataset
    print("If path is existed:", os.path.exists(dataset_path))
    # Load data
    X, y = utils.load_and_process_data(dataset_path, features_to_remove=None)

    # Split the dataset into 70% training and 30% testing sets
    X_train, X_test, y_train, y_test = utils.split_data(X, y, test_size=0.3, val_size=0.2, random_state=42)
    if(cfg.wandb_on_off and cfg.name == "Project_midterm"):
        wandb.init(project="Project_midterm")

    # enginner Tourque and Tool wear features to train the model according to the correlation matrix
    X_train_f = X_train[:,3:]
    X_test_f = X_test[:,3:]

    l = 0
    p = 0
    lr = 0
    mlp = 1

    if l == 1:
        # Linear Regression
        print("Linear Regression training")
        model_L = LinearRegression(n_feature=X_train_f.shape[1], epoch = cfg.epoch, lr = cfg.lr_linear_regression, batch_size=cfg.batch_size, gd = cfg.gd)
        model_L.fit(X_train_f, y_train)
        metrics_L = model_L._evaluate(X_test_f, y_test)
        print(f"Linear Regression evaluation: {metrics_L}")

    if p == 1:
        # Perceptron
        print("Perceptron training")
        y_train_P = y_train.copy()
        y_test_P = y_test.copy()
        y_train_P[y_train_P == 0] = -1
        y_test_P[y_test_P == 0] = -1
        model_P = Perceptron(n_feature=X_train_f.shape[1], epoch=cfg.epoch, lr=cfg.lr, tol=cfg.tol, wandb=cfg.wandb_on_off, gd=cfg.gd)
        model_P.fit(X_train_f, y_train_P)
        metrics_P = model_P.evaluate(X_test_f, y_test_P)
        print(f"Perceptron evaluation: {metrics_P}")

    if lr == 1:
        # Logistic Regression
        print("Logistic Regression training")
        model_LR = LogisticRegression(n_feature=X_train_f.shape[1], epoch=cfg.epoch, lr=cfg.lr, tol=cfg.tol, wandb=cfg.wandb_on_off, gd=cfg.gd)
        model_LR.fit(X_train_f, y_train)
        metrics_LR = model_LR._evaluate(X_test_f, y_test)
        print(f"Logistic Regression evaluation: {metrics_LR}")

    if mlp == 1:
        # MLP
        print("MLP training")
        input_size = X_train_f.shape[1]
        layers_list = [input_size] + cfg.hidden_layers + [1]
        model_MLP = MLP(layers_list)
        model_MLP.train(X_train_f, y_train, cfg.epoch, cfg.lr, cfg.batch_size, cfg.gd)
        metrics_MLP = model_MLP.evaluate(X_test_f, y_test)

        input_size1 = X_train[:,0:3].shape[1]
        layers_list1 = [input_size1] + cfg.hidden_layers + [1]
        model_MLP1 = MLP(layers_list1)
        model_MLP1.train(X_train[:,0:3], y_train, cfg.epoch, cfg.lr, cfg.batch_size, cfg.gd)
        metrics_MLP1 = model_MLP.evaluate(X_test_f, y_test)
       
        input_size2 = X_train.shape[1]
        layers_list2 = [input_size2] + cfg.hidden_layers + [1]
        model_MLP2 = MLP(layers_list2)
        model_MLP2.train(X_train, y_train, cfg.epoch, cfg.lr, cfg.batch_size, cfg.gd)
        metrics_MLP2 = model_MLP.evaluate(X_test_f, y_test)
 
        print(f"MLP evaluation with two correlated features: {metrics_MLP}")
        print(f"MLP evaluation with another three features: {metrics_MLP1}")
        print(f"MLP evaluation with all features: {metrics_MLP2}")

    # print evaluation
    print(f"Linear Regression evaluation: {metrics_L}")
    print(f"Perceptron evaluation: {metrics_P}")
    print(f"Logistic Regression evaluation: {metrics_LR}")
    print(f"MLP evaluation: {metrics_MLP}")


if __name__ == "__main__":
    main()