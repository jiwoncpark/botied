from tqdm import tqdm
import numpy as np
import torch
from botorch.utils.sampling import draw_sobol_samples
from .base_acquisition import BaseAcquisition
import logging
logger = logging.getLogger(__name__)


class Random(BaseAcquisition):

    def __init__(self):
        super().__init__()

    def get_scores_from_batch(
            self,
            f_cand: torch.Tensor,
            f_baseline: torch.Tensor,
            *args, **kwargs) -> torch.Tensor:
        """Evaluate the scores on a batch

        Parameters
        ----------
        f_cand : torch.tensor
            Candidate f of shape [n_mc_samples, num_candidates, f_dim]
        f_baseline : torch.tensor
            Non-dominated, known f of shape [n_mc_samples, num_nd, f_dim]

        Returns
        -------
        torch.Tensor
            scores

        """
        pass

    def select_subset(
            self,
            f_cand: torch.tensor,
            q_batch_size: int,
            *args, **kwargs) -> torch.Tensor:
        """Select a batch sequence by sequence, based on pre-fetched oracle
        samples

        Parameters
        ----------
        f_cand : torch.tensor
            Candidate f of shape [n_mc_samples, num_candidates, f_dim]
        q_batch_size : int
            How many candidates to select

        Returns
        -------
        dict
            selected_idx : list
            selected_scores : list

        """
        # rng = np.random.default_rng(seed)
        num_cand = f_cand.shape[1]
        selected_idx = np.random.choice(
            num_cand, size=q_batch_size, replace=False)
        return dict(selected_idx=selected_idx,
                    selected_scores=np.zeros(num_cand),)

    def optimize(
            self, bounds, q_batch_size,
            *args, **kwargs):
        """Draw Sobol samples """
        dim_x = len(bounds[0])
        standard_bounds = torch.tensor([[0.0]*dim_x, [1.0]*dim_x])
        standard_bounds = standard_bounds.to(
            device=bounds.device, dtype=bounds.dtype)
        new_x = draw_sobol_samples(
            bounds=standard_bounds, n=q_batch_size, q=1).squeeze(1)
        # [n, q, f_dim] -> [n, f_dim]
        return {'new_x': new_x}


