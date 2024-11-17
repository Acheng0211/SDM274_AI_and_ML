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
    X, y = utils.load_and_process_data(dataset_path, features_to_remove=None) #ok
    # print(X.shape, y.shape)
    X_train, X_test, y_train, y_test = utils.split_data(X[:,1:], y, test_size=0.3, val_size=0.2, random_state=42)
    # np.savetxt("compare.csv",  y, delimiter=", ")

    if(cfg.wandb_on_off and cfg.name == "Project_midterm"):
        wandb.init(project="Project_midterm")

    # 不同的模型需要按照模型特点喂入不同的训练集，参考作业和md文档

    # Linear Regression
    X_train_L = X_train[:,2].reshape(-1,1)
    X_test_L = X_test[:,2].reshape(-1,1)
    model_L = LinearRegression(n_feature=X_train_L.shape[1], epoch = cfg.epoch, lr = cfg.lr, gd = cfg.gd)
    model_L.fit(X_train_L, y_train)
    metrics_L = model_L._evaluate(X_test_L, y_test)
    print(f"L evaluation: {metrics_L}")

    # # Perceptron
    # y_train_P = y_train.copy()
    # y_test_P = y_test.copy()
    # y_train_P[y_train_P == 0] = -1
    # y_test_P[y_test_P == 0] = -1
    # model_P = Perceptron(n_feature=X_train.shape[1], epoch=cfg.epoch, lr=cfg.lr, tol=cfg.tol, wandb=cfg.wandb_on_off, gd=cfg.gd)
    # model_P.fit(X_train, y_train_P)
    # metrics_P = model_P._evaluate(X_test, y_test_P)
    # print(f"P evaluation: {metrics_P}")


    # # Logistic Regression
    # y_train_LR = y_train.copy()
    # y_test_LR = y_test.copy()
    # model_LR = LogisticRegression(n_feature=X_train.shape[1], epoch=cfg.epoch, lr=cfg.lr, tol=cfg.tol, wandb=cfg.wandb_on_off, gd=cfg.gd)
    # model_LR.fit(X_train, y_train_LR)
    # metrics_LR = model_LR._evaluate(X_test, y_test_LR)
    # print(f"LR evaluation: {metrics_LR}")

    # # MLP
    # input_size = X_train.shape[1] - 1
    # layers_list = [input_size, 10, 1]
    # model_MLP = MLP(layers_list)
    # metrics_MLP = utils.cross_validate(model_MLP, X[:,1:], y, cfg.k, cfg.epoch, cfg.lr, cfg.batch_size, cfg.gd)
    # print(f"MLP evaluation: {metrics_MLP}")
    # utils.plot_loss(model_MLP, layers_list, None, cfg.gd)

if __name__ == "__main__":
    main()