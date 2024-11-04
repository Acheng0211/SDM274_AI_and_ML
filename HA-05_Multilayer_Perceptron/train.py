import hydra
import hydra.conf
import wandb
from omegaconf import DictConfig
import utils
from model import MLP
import os
import matplotlib.pyplot as plt


@hydra.main(version_base="1.3", config_path="./conf", config_name="config_MLP")
def main(cfg: DictConfig):

    # Preprocess dataset
    dataset_path = cfg.dataset_path
    print("If path is existed:", os.path.exists(dataset_path))
    data_raw = utils.load_data(dataset_path)
    X, y = utils.classify_data(data_raw)
    X_train, X_val, X_test, y_train, y_val, y_test = utils.split_data(X, y, test_size=0.3, val_size=0.2, random_state=42)
    
    if(cfg.wandb_on_off and cfg.name == "MLP"):
        if cfg.gd == "SGD":
            wandb.init(project="MLP_SGD")
        elif cfg.gd == "MBGD":
            wandb.init(project="MLP_MBGD")
        model = MLP(input_size=cfg.input_size, batch_size=cfg.batch_size, num_classes=cfg.num_classes, epoch=cfg.epoch, gd=cfg.gd, wandb=cfg.wandb_on_off, lossf=cfg.loss_function, tol=cfg.tol, lr=cfg.lr, activation=cfg.activation_function)

    model.fit(X_train, y_train)
    model.evaluate(X_test, y_test)
    model.plot_loss(model.loss, cfg.name)
    model.plot_data(X_test, y_test)
    wandb.finish()

if __name__ == "__main__":
    main()
    # evaluation results: accuracy: 0.717948717948718, recall: 0.45, precision: 1.0, F1: 0.6206896551724138