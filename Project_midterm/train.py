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
    X, y = utils.load_and_process_data(dataset_path, features_to_remove=None)
    print(X.shape, y.shape)
    # X_train, X_test, y_train, y_test = utils.split_data(X, y, test_size=0.3, val_size=0.2, random_state=42)
    np.savetxt("compare.txt", np.concatenate((X.reshape(-1, 1), y), axis=1), delimiter=", ")

    if(cfg.wandb_on_off and cfg.name == "Project_midterm"):
        wandb.init(project="Project_midterm")

    # 不同的模型需要按照模型特点喂入不同的训练集，参考作业和md文档

    # Linear Regression
    model_L = LinearRegression(n_feature=X_train.shape[1], epoch = cfg.epoch, lr = cfg.lr, gd = cfg.gd)
    model_L.fit(X_train, y_train)
    model_L.evaluate(X_test, y_test)

    # Perceptron
    model_P = Perceptron(n_feature=X_train.shape[1], epoch=cfg.epoch, lr=cfg.lr, tol=cfg.tol, wandb=cfg.wandb_on_off, gd=cfg.gd)
    model_P.fit(X_train, y_train)
    model_P.evaluate(X_test, y_test)

    # Logistic Regression
    model_LR = LogisticRegression(n_feature=X_train.shape[1], epoch=cfg.epoch, lr=cfg.lr, tol=cfg.tol, wandb=cfg.wandb_on_off, gd=cfg.gd)
    model_LR.fit(X_train, y_train)
    model_LR.evaluate(X_test, y_test)

    # MLP
    model_MLP = MLP(cfg.layer_list)
    metrics = utils.cross_validate(model_MLP, X, y, cfg.k, cfg.epoch, cfg.lr, cfg.batch_size, cfg.gd)


if __name__ == "__main__":
    main()