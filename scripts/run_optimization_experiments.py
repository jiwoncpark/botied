import pyvinecopulib as pv  # need to import first, glibc issues
import os
import torch
from botied import ExperimentManager
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import wandb
import logging
logger = logging.getLogger(__name__)
torch.set_default_tensor_type(torch.DoubleTensor)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    wandb.config = OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True)
    wandb.config['cwd'] = os.getcwd()
    logger.info("Working directory : {}".format(os.getcwd()))
    wandb.init(
        **cfg.wandb,
        config=wandb.config,
        settings=wandb.Settings(start_method="thread")
    )
    dtype = torch.double
    device = torch.device(cfg.device_name)
    # Define objective
    objective = instantiate(cfg.objective)
    objective.set_tkwargs(device=device, dtype=dtype)
    # Define model wrapper
    model_wrapper = instantiate(cfg.model_wrapper)
    model_wrapper.set_tkwargs(device=device, dtype=dtype)
    # Define main API
    cdf_indicator = instantiate(cfg.cdf_indicator)
    botied_obj = ExperimentManager(
        objective, model_wrapper,
        acquisitions_kwargs=cfg.acquisitions,
        cdf_indicator=cdf_indicator,
        dtype=dtype, device=device)
    botied_obj.optimize_over_design_space(**cfg.run_kwargs)
    wandb.finish()


if __name__ == '__main__':
    import sys
    sys.argv.append('hydra.job.chdir=True')
    main()
