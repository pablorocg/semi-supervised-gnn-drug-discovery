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

    # Get dataset properties
    n_outputs = dm.num_tasks
    task_type = dm.task_type
    in_channels = dm.num_features
    

    # model = hydra.utils.instantiate(cfg.model.init)

    # Instantiate model
    model = instantiate(
        cfg.model.init,
        in_channels=in_channels,
        out_channels=n_outputs,
    )

    # Create lightning module
    lightning_module = instantiate(
        cfg.lightning_module.init,
        _target_=BaselineModule,
        model=model,
        num_outputs=n_outputs,
        task_type=task_type,
  
    )



if __name__ == "__main__":
    main()