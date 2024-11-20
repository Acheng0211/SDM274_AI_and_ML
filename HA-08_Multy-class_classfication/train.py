import hydra
import hydra.conf
import wandb
from omegaconf import DictConfig
import utils
from model import MLP


@hydra.main(version_base="1.3", config_path="./conf", config_name="config_MLP")
def main(cfg: DictConfig):
    # generate data, initialize outputs of models and evaluations
    X, y, y_raw = utils.generate_data(cfg.mission)
    models = []
    results = []
    
    # train and evaluate
    for layers in cfg.layers_list:
        model = MLP(layers)
        metrics = utils.cross_validate(model, X, y, cfg.k, cfg.epoch, cfg.lr, cfg.batch_size, cfg.gd)
        models.append(model)
        results.append((layers,metrics))

    # plot figure and loss
    if cfg.mission == "Nonlinear":
        utils.plot_nonliear(X, y_raw, models, cfg.layers_list, cfg.gd)
    elif cfg.mission == "Classifier":
        utils.plot_decision_boundary(X, y, models, cfg.layers_list, cfg.gd)
    utils.plot_loss(models, cfg.layers_list, cfg.mission, cfg.gd)

    # display results
    utils.LOG_RESULTS(results, cfg.gd)

if __name__ == "__main__":
    main()