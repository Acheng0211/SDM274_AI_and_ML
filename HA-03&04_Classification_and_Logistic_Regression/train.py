import hydra
import wandb
from omegaconf import DictConfig
import utils as uts
from model import Perceptron


@hydra.main(version_base="1.3", config_path="./conf", config_name="config")
def main(cfg: DictConfig):
    if(cfg.wandb_on_off):
        wandb.init(project="hw3", config=cfg)

    dataset_path = cfg.dataset_path
    print("preprocess dataset ")
    data_raw = uts.load_data(dataset_path)
    data_filtered = uts.filter(data_raw)
    X, y = uts.classify_data(data_filtered)

    X_train, X_test, y_train, y_test = uts.split_data(X, y, test_size=0.3, random_state=42)
    
    model = Perceptron(n_feature=X_train.shape[1], epoch=cfg.epoch, lr=cfg.lr, tol=cfg.tol, wandb=cfg.wandb_on_off, gd=cfg.gd)
    model.fit(X_train, y_train)

    model.evaluate(X_test, y_test)

if __name__ == "__main__":
    main()