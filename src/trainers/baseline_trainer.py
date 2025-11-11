from hydra.utils import instantiate
from src.data.moleculenet import MoleculeNetDataModule
from src.data.qm9 import QM9DataModule
from src.lightning_modules.baseline import BaselineModule




from itertools import chain
import hydra
import torch
from omegaconf import OmegaConf

from src.utils.utils import seed_everything
from src.utils.path_utils import get_configs_dir


@hydra.main(
    config_path=get_configs_dir(),
    config_name="baseline_config.yaml",
    version_base=None,
)
def main(cfg):
    # print out the full config
    print(OmegaConf.to_yaml(cfg))

    seed_everything(cfg.seed, cfg.force_deterministic)


    # Create an instance of the wandblogger and initialize it
    
    # logger = hydra.utils.instantiate(cfg.logger)
    # hparams = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    # logger.init_run(hparams)

    # dm = hydra.utils.instantiate(cfg.dataset.init)

    # model = hydra.utils.instantiate(cfg.model.init)

    # trainer = hydra.utils.instantiate(cfg.trainer.init, models=models, logger=logger, datamodule=dm, device=device)

    # results = trainer.train(**cfg.trainer.train)
    # results = torch.Tensor(results)



if __name__ == "__main__":
    main()