"""Base class for acquisition function

"""
from abc import ABC, abstractmethod
from tqdm import tqdm
from collections.abc import Iterable
import numpy as np
import torch
from botorch.sampling.normal import SobolQMCNormalSampler
import logging
logger = logging.getLogger(__name__)


class BaseAcquisition(ABC):
    """Abstract base class representing an acquisition function"""

    @abstractmethod
    def __init__(self):
        self.raw_samples = 512  # used for intialization heuristic
        self.num_restarts = 10
        self.batch_limit = 5
        self.maxiter = 200
        self.mc_samples = 128
        self.sampler = SobolQMCNormalSampler(
            sample_shape=torch.Size([self.mc_samples]))

    def validate_shapes(self, y_pool, y_known):
        """
        Parameters
        ----------
        y_pool : torch.tensor
            Candidate f of shape [num_mc_samples, num_candidates, f_dim]
        y_known : torch.tensor
            Known f of shape [n_baseline_samples, num_known, f_dim]

        """
        if y_known.dim() == 2:
            y_known = y_known.unsqueeze(0)  # n_baseline_samples = 1
        n_mc_samples_candidates, n_candidates, f_dim = y_pool.shape
        n_mc_samples_known, n_known, f_dim_baseline = y_known.shape
        if n_mc_samples_known == 1:
            logger.info("Reduced to non-noisy version.")
        if n_mc_samples_known != n_mc_samples_candidates:
            raise NotImplementedError(
                "Candidate and known inference results are assumed to be"
                " aligned.")
        if f_dim != f_dim_baseline:
            raise ValueError(
                "Candidate and known objective dimensions must be the same.")

    def select_from_dataset(
        self,
        mol_ids: Iterable,
        f_cand: torch.tensor,
        q_batch_size: int,
        *args,
        **kwargs) -> tuple:
        """Select a batch when given access to the entire dataset in memory

        Parameters
        ----------
        f_cand : torch.tensor
            Candidate f of shape [n_mc_samples, num_candidates, f_dim]
        f_baseline_nd : torch.tensor
            Non-dominated, known f of shape [n_mc_samples, num_nd, f_dim]

        Returns
        -------
        selected_idx : list
            Indices of selected candidates from `y_pool`, of length
            `q_batch_size`
        initial_nd : torch.Tensor
        final_nd : torch.Tensor
        scores_selected : list

        """
        scores = self.get_scores_from_batch(f_cand, *args, **kwargs)
        scores_selected, idx_selected = torch.topk(
            scores, k=q_batch_size, dim=0, sorted=True
        )
        mol_ids_selected = mol_ids[idx_selected]
        return dict(
            mol_ids_selected=mol_ids_selected,
            scores_selected=scores_selected,)

    def select_subset(
            self,
            f_cand: torch.tensor,
            f_baseline: torch.tensor,
            q_batch_size: int,
            *args, **kwargs) -> torch.Tensor:
        """Select a batch sequence by sequence, based on pre-fetched oracle
        samples

        Parameters
        ----------
        f_cand : torch.tensor
            Candidate f of shape [n_mc_samples, num_candidates, f_dim]
        f_baseline : torch.tensor
            Known f of shape [n_mc_samples, num_known, f_dim]
        q_batch_size : int
            How many candidates to select

        Returns
        -------
        dict
            selected_idx : list
            selected_scores : list

        """
        # Validate shapes
        self.validate_shapes(f_cand, f_baseline)
        # Handle simple cases
        n_candidates = f_cand.shape[1]
        if q_batch_size >= n_candidates:
            logger.info("Selecting all of candidate pool")
            q_batch_size = n_candidates

        # Init
        selected_idx, f_selected, selected_scores = [], [], []

        # Select candidates sequentially
        for _ in tqdm(range(q_batch_size), desc=self.__class__.__name__):
            # Init best for this batch
            max_idx = None
            max_score = -1
            latest_selected_f = None

            scores = self.get_scores_from_batch(
                f_cand=f_cand,  # [n_mc_samples, batch_size, f_dim]
                f_baseline=f_baseline,  # [n_mc_samples, num_known, f_dim]
            )  # [batch_size,]
            # Exclude already selected before getting max in batch
            exclude = torch.from_numpy(
                np.isin(np.arange(n_candidates), selected_idx))
            scores[exclude] = -1.0
            # Find best in batch
            max_idx = scores.argmax().item()
            max_score = scores[max_idx].item()
            latest_selected_f = f_cand[:, max_idx, :]

            # Append selected candidate to solution
            selected_idx.append(max_idx)
            selected_scores.append(max_score)
            f_selected.append(latest_selected_f)

        # Note jwp: joint score computation skipped
        # TODO jwp: check if joint score useful to copy across selected rows
        return dict(selected_idx=selected_idx,
                    selected_scores=selected_scores,)
