import numpy as np
import torch
from botorch.utils import draw_sobol_samples
from botorch.utils.multi_objective.box_decompositions.dominated import (
    DominatedPartitioning,
)
from botorch.utils.multi_objective.pareto import is_non_dominated
import logging
logger = logging.getLogger(__name__)


def get_non_dominated_mask(Y):
    Y = torch.tensor(Y).double()
    return is_non_dominated(Y).cpu().numpy()


def get_hv(y, ref_point):
    bd = DominatedPartitioning(ref_point.double(), y)
    hv = bd.compute_hypervolume().item()
    return hv


def get_max_hv(prob, num_samples=10000):
    try:
        max_hv = prob.max_hv
    except:
        logger.info("Approximating max hv...")
        max_hv = approximate_max_hv(prob, num_samples=num_samples)
    return max_hv


def approximate_max_hv(prob, num_samples=10000):
    x = draw_sobol_samples(prob.bounds, n=num_samples, q=1).squeeze(1)
    y = prob(x)
    bd = DominatedPartitioning(prob.ref_point.double(), y)
    return bd.compute_hypervolume().item()


def get_log_hv_diff(hv, max_hv):
    return np.log(max_hv - np.asarray(hv))
