if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
import shutil
from vla_project.workspace.base_workspace import BaseWorkspace
from vla_project.policy.TransfusionImgActPolicy import TransfusionImgActPolicy
from vla_project.datasets.base_dataset import BaseImageDataset
from vla_project.utils.lr_scheduler import get_scheduler
from vla_project.models.diffusion.ema_model import EMAModel
from vla_project.envs.env_runner.base_image_runner import BaseImageRunner
from vla_project.utils.checkpoint_util import TopKCheckpointManager
from vla_project.utils.pytorch_util import optimizer_to, dict_apply
from vla_project.utils.json_logger import JsonLogger

# from vla_project.models.llama2c import 
# from vla_project.models.transfusion import Transfussion
# from vla_project.utils.diffusion_utils import DiffusionUtils




OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainTransfusionImgActWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch']
    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        # self.model: TransfusionImgActPolicy = hydra.utils.instantiate(cfg.policy)

        # self.ema_model: TransfusionImgActPolicy = None
        # if cfg.training.use_ema:
        #     self.ema_model = copy.deepcopy(self.model)

        # # configure training state
        # self.optimizer = self.model.get_optimizer(**cfg.optimizer)

        # configure training state
        self.global_step = 0
        self.epoch = 0

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)
        
        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)



@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("configs")), 
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainTransfusionImgActWorkspace(cfg)
    workspace.run()

if __name__ == "__main__":
    main()

