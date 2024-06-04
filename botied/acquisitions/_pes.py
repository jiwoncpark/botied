from tqdm import tqdm
import numpy as np
import torch
from botorch.optim.optimize import optimize_acqf
from .base_acquisition import BaseAcquisition
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.acquisition.multi_objective.utils import (
    sample_optimal_points,
    random_search_optimizer,
    compute_sample_box_decomposition
)
from botorch.acquisition.multi_objective.predictive_entropy_search import (
    qMultiObjectivePredictiveEntropySearch,
)
from botorch.utils.transforms import unnormalize, normalize
from botorch.utils.sampling import draw_sobol_samples
import logging
logger = logging.getLogger(__name__)


def _pes_callback(
        f_cand, f_baseline, x_cand, x_baseline, model,
        *args, **kwargs):
    """Wrapper around `qMultiObjectivePredictiveEntropySearch`
    to access acquisition values directly from samples

    Parameters
    ----------
    f_cand : torch.tensor
        Query candidates from a sample path, of shape `[n_candidates, 1, f_dim]`
    f_baseline : torch.tensor
        Prev evaluated points from a sample path, of shape `[num_baseline, f_dim]`
    model : Model
        Fitted surrogate

    Returns
    -------
    torch.tensor
        Acquisition values for the query candidates, of shape `[n_candidates,]`

    """
    # Find non-dominated points among f_cand
    # We first need MC samples of the optimal inputs and outputs
    # In this subselection setting, we can only find optimal
    optimal_pf_samples = torch.cat(
        [f_cand.squeeze(1), f_baseline], dim=0)  # [num_cand + num_baseline, f_dim]
    optimal_ps_samples = torch.cat(
        [x_cand, x_baseline], dim=0)  # [num_cand + num_baseline, input_dim]
    is_nd = is_non_dominated(optimal_pf_samples)
    optimal_ps_samples = optimal_ps_samples[is_nd]
    pes_obj = qMultiObjectivePredictiveEntropySearch(
        model=model,
        pareto_sets=optimal_ps_samples.unsqueeze(0)
    )
    scores = pes_obj(x_cand.unsqueeze(-2))
    return scores


class PES(BaseAcquisition):

    def __init__(
            self,
            num_pareto_samples: int = 10,
            num_pareto_points: int = 10,
            pop_size: int = 2000,
            max_tries: int = 10):
        # Only used in optimization setting
        self.num_pareto_samples = num_pareto_samples
        self.num_pareto_points = num_pareto_points
        self.pop_size = pop_size
        self.max_tries = max_tries
        # We set the parameters for the random search
        self.optimizer_kwargs = {
            "pop_size": self.pop_size,
            "max_tries": self.max_tries,
        }
        super().__init__()

    def get_scores_from_batch(
            self,
            f_cand: torch.Tensor,
            f_baseline: torch.Tensor,
            x_cand: torch.Tensor,
            x_baseline: torch.Tensor,
            model,  # TODO: add typing
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
            scores_i = _pes_callback(
                f_cand=f_cand[sample_i].unsqueeze(-2),
                # ~ [n_candidates, q=1, f_dim]
                f_baseline=f_baseline[sample_i],
                model=model,
                x_cand=x_cand,
                x_baseline=x_baseline,
            )
            # Update running mean
            scores += (scores_i - scores)/(sample_i+1)
        return scores

    def select_subset(
            self,
            f_cand: torch.tensor,
            f_baseline: torch.tensor,
            q_batch_size: int,
            x_cand: torch.tensor,
            x_baseline: torch.tensor,
            model,
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
                f_cand=f_cand,  # [n_mc_samples, num_candidates, f_dim]
                f_baseline=f_baseline,  # [n_mc_samples, num_baseline, f_dim]
                x_cand=x_cand,
                x_baseline=x_baseline,
                model=model,
            )  # [batch_size,]
            # Exclude already selected before getting max in batch
            exclude = torch.from_numpy(
                np.isin(np.arange(n_candidates), selected_idx))
            scores[exclude] = -np.inf  # acq value can be negative for PES
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
            self, model, bounds, q_batch_size,
            *args, **kwargs):
        """Samples a set of random weights for each candidate in the batch,
        performs sequential greedy optimization
        of the PES acquisition function, and returns a new candidate and
        observation."""
        total_retries = 100
        for trial_i in range(1, total_retries+1):
            try:
                ps, _ = sample_optimal_points(
                    model=model,
                    bounds=bounds,
                    num_samples=self.num_pareto_samples,
                    num_points=self.num_pareto_points,
                    optimizer=random_search_optimizer,
                    optimizer_kwargs=self.optimizer_kwargs,
                )
                pes = qMultiObjectivePredictiveEntropySearch(
                    model=model, pareto_sets=ps)
                # Sequentially greedy optimization
                new_x, _ = optimize_acqf(
                    acq_function=pes,
                    bounds=bounds,
                    q=q_batch_size,
                    num_restarts=self.num_restarts,
                    raw_samples=self.raw_samples,
                    sequential=True,
                )
                break
            except:
                logger.info(f"Retrying optimization, trial {trial_i+1}...")
                pass
        if trial_i == total_retries:
            return {'new_x': draw_sobol_samples(bounds, n=1, q=1).squeeze(1)}
        new_x = new_x.detach()
        new_x = normalize(new_x, bounds)
        return {'new_x': new_x}
