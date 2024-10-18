import hydra
import hydra.conf
import wandb
from omegaconf import DictConfig
import utils as uts
from model import Perceptron
from model import LogisticRegression
import os


@hydra.main(version_base="1.3", config_path="./conf", config_name="config_LogisticRegression")
def main(cfg: DictConfig):

    # Preprocess dataset
    dataset_path = cfg.dataset_path
    print("If path is existed:", os.path.exists(dataset_path))
    data_raw = uts.load_data(dataset_path)
    data_filtered = uts.filter(data_raw)
    X, y = uts.classify_data(data_filtered)
    X_train, X_test, y_train, y_test = uts.split_data(X, y, test_size=0.3, random_state=42)
    
    if(cfg.wandb_on_off and cfg.name == "Perceptron"):
        wandb.init(project="Perceptron")
        # Train model
        model = Perceptron(n_feature=X_train.shape[1], epoch=cfg.epoch, lr=cfg.lr, tol=cfg.tol, wandb=cfg.wandb_on_off, gd=cfg.gd)
    elif(cfg.wandb_on_off and cfg.name == "LogisticRegression"):
        if cfg.gd == "SGD":
            wandb.init(project="Logistic_Regression_SGD")
        elif cfg.gd == "MBGD":
            wandb.init(project="Logistic_Regression_MBGD")
        model = LogisticRegression(n_feature=X_train.shape[1], epoch=cfg.epoch, lr=cfg.lr, tol=cfg.tol, wandb=cfg.wandb_on_off, gd=cfg.gd)

    # Fit and evaluate
    model.fit(X_train, y_train)
    model.evaluate(X_test, y_test)

if __name__ == "__main__":
    main()
    # evaluation results: accuracy: 0.717948717948718, recall: 0.45, precision: 1.0, F1: 0.6206896551724138