from abc import ABC, abstractmethod
from typing import Optional
import torch
from omegaconf import OmegaConf
from hydra.utils import instantiate
from botorch.utils.sampling import draw_sobol_samples
import logging
logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Base model class

    """
    def __init__(self):
        pass

    def set_tkwargs(self, device, dtype):
        self.device = device
        self.dtype = dtype

    @abstractmethod
    def fit_model(self):
        raise NotImplementedError

    @abstractmethod
    def sample_predictions(self):
        raise NotImplementedError
