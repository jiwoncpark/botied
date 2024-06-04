from tqdm import tqdm
import numpy as np
import torch
from botorch.acquisition.objective import GenericMCObjective
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.utils.multi_objective.scalarization import (
    get_chebyshev_scalarization)
from botorch.optim.optimize import optimize_acqf_list
from botorch.utils.sampling import sample_simplex
from botorch.utils.transforms import unnormalize, normalize
from .base_acquisition import BaseAcquisition
import logging
logger = logging.getLogger(__name__)


def _qparego_callback(f_cand, f_baseline):
    """Deconstructed `qParEGO` to access pointwise improvement

    Parameters
    ----------
    f_cand : torch.tensor
        Query candidates, of shape `[n_candidates, 1, f_dim]`
    f_baseline : torch.tensor
        Known points, of shape `[num_baseline, f_dim]`

    Returns
    -------
    torch.tensor
        Improvement values for the query candidates, of shape `[n_candidates,]`

    """
    f_dim = f_cand.shape[-1]
    weights = sample_simplex(f_dim).squeeze().to(
        device=f_cand.device, dtype=f_cand.dtype)  # [f_di,]
    objective_obj = GenericMCObjective(
        get_chebyshev_scalarization(weights=weights, Y=f_baseline))
    cand_obj = objective_obj(f_cand.squeeze(-2))  # [n_candidates]
    known_obj = objective_obj(f_baseline)  # [num_baseline,]
    diff = cand_obj - known_obj.max(dim=-1, keepdim=True).values  # [n_candidates,]
    improvement = diff.clamp_min(0) # [n_candidates,]
    return improvement


class qNParEGO(BaseAcquisition):

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
        n_mc_samples = f_cand.shape[0]
        scores = 0.0
        for sample_i in range(n_mc_samples):
            scores_i = _qparego_callback(
                f_cand=f_cand[sample_i].unsqueeze(-2),
                # ~ [n_candidates, q=1, f_dim]
                f_baseline=f_baseline[sample_i],
            )
            # Update running mean
            scores += (scores_i - scores)/(sample_i+1)
        return scores

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

    def optimize(
            self, model, train_x, bounds, q_batch_size,
            *args, **kwargs):
        """Samples a set of random weights for each candidate in the batch,
        performs sequential greedy optimization
        of the qNParEGO acquisition function, and returns a new candidate and
        observation."""
        with torch.no_grad():
            pred = model.posterior(train_x).mean
        acq_func_list = []
        for _ in range(q_batch_size):
            weights = sample_simplex(
                model._num_outputs,
                dtype=train_x.dtype, device=train_x.device).squeeze()
            objective = GenericMCObjective(
                get_chebyshev_scalarization(weights=weights, Y=pred)
            )
            acq_func = qNoisyExpectedImprovement(
                model=model,
                objective=objective,
                X_baseline=train_x,
                sampler=self.sampler,
                prune_baseline=True,
            )
            acq_func_list.append(acq_func)
        # optimize
        new_x, _ = optimize_acqf_list(
            acq_function_list=acq_func_list,
            bounds=bounds,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,  # used for intialization heuristic
            options={"batch_limit": self.batch_limit, "maxiter": self.maxiter},
        )
        new_x = new_x.detach()
        new_x = normalize(new_x, bounds)
        return {'new_x': new_x}


