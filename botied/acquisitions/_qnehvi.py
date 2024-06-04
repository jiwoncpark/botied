from tqdm import tqdm
import numpy as np
import torch
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.box_decompositions \
    import FastNondominatedPartitioning
from botorch.acquisition.multi_objective.monte_carlo import (
    qExpectedHypervolumeImprovement,
    qNoisyExpectedHypervolumeImprovement,
)
from copulala.acquisitions.utils import get_dummy_gp
from botorch.optim.optimize import optimize_acqf, optimize_acqf_list
from botorch.utils.transforms import unnormalize, normalize
from .base_acquisition import BaseAcquisition
import logging
logger = logging.getLogger(__name__)


def _qehvi_callback(f_cand, f_baseline, ref_point):
    """Wrapper around `qExpectedHypervolumeImprovement` to access acquisition
    values directly from samples

    Parameters
    ----------
    f_baseline : torch.tensor
        Known points that are non-dominated, of shape `[num_baseline, f_dim]`
    f_cand : torch.tensor
        Query candidates, of shape `[n_mc_samples, n_candidates, 1, f_dim]`
    ref_point : torch.tensor
        Reference point, of shape `[f_dim,]`

    Returns
    -------
    torch.tensor
        Acquisition values for the query candidates, of shape `[n_candidates,]`

    """
    acq_fn = qExpectedHypervolumeImprovement(
        model=get_dummy_gp(),  # not used
        ref_point=ref_point,
        partitioning=FastNondominatedPartitioning(
            ref_point=ref_point, Y=f_baseline),
    )
    return acq_fn._compute_qehvi(f_cand)


class qNEHVI(BaseAcquisition):

    def __init__(self):
        super().__init__()

    def get_scores_from_batch(
            self,
            f_cand: torch.Tensor,
            f_baseline: torch.Tensor,
            ref_point: torch.Tensor,
            *args, **kwargs) -> torch.Tensor:
        """Evaluate the scores on a batch

        Parameters
        ----------
        f_cand : torch.tensor
            Candidate f of shape [n_mc_samples, num_candidates, f_dim]
        f_baseline : torch.tensor
            Prev evaluated points' f of shape [n_mc_samples, num_baseline, f_dim]

        Returns
        -------
        torch.Tensor
            scores

        """
        n_mc_samples = f_cand.shape[0]
        scores = 0.0
        # TODO jwp: speed up
        for sample_i in range(n_mc_samples):
            f_baseline_nd = f_baseline[sample_i][
                is_non_dominated(f_baseline[sample_i])]  # [num_nd, f_dim]
            # When used for qnehvi, this callback simply evaluates HVI
            # on a sample level
            scores_i = _qehvi_callback(
                f_cand=f_cand[[sample_i]].unsqueeze(-2),
                # ~ [num_samples=1, n_candidates, q=1, f_dim]
                f_baseline=f_baseline_nd, # [num_nd, f_dim]
                ref_point=ref_point,
            )
            # Update running mean
            scores += (scores_i - scores)/(sample_i+1)
        return scores

    def select_subset(
            self,
            f_cand: torch.tensor,
            f_baseline: torch.tensor,
            q_batch_size: int,
            ref_point: torch.tensor,
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
                f_cand=f_cand,  # [n_mc_samples, num_cand, f_dim]
                f_baseline=f_baseline,  # [n_mc_samples, num_known, f_dim]
                ref_point=ref_point,
            )  # [batch_size,]
            # Exclude already selected before getting max in batch
            exclude = torch.from_numpy(
                np.isin(np.arange(n_candidates), selected_idx))
            scores[exclude] = -1.0
            # Find best in batch
            max_idx = scores.argmax().item()
            max_score = scores[max_idx].item()
            latest_selected_f = f_cand[:, max_idx, :]
            # Update baseline set to include posterior mean of selected candidate
            f_baseline = torch.cat(
                [
                    f_baseline,  # [num_samples, num_baseline, f_dim]
                    # Take arbitrary subset from available posterior samples
                    latest_selected_f.unsqueeze(1),
                    # ~ [num_samples, 1, f_dim]
                ],
                dim=1
            )  # [num_samples, num_baseline+(step_idx+1), f_dim]

            # Append selected candidate to solution
            selected_idx.append(max_idx)
            selected_scores.append(max_score)
            f_selected.append(latest_selected_f)

        # Note jwp: joint score computation skipped
        # TODO jwp: check if joint score useful to copy across selected rows
        return dict(selected_idx=selected_idx,
                    selected_scores=selected_scores,)

    def optimize(
            self, model, train_x, bounds, q_batch_size, ref_point,
            *args, **kwargs):
        """Samples a set of random weights for each candidate in the batch,
        performs sequential greedy optimization
        of the qNParEGO acquisition function, and returns a new candidate and
        observation."""
        # partition non-dominated space into disjoint rectangles
        acq_func = qNoisyExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point.tolist(),  # use known reference point
            X_baseline=train_x,
            # prune baseline points that have estimated zero probability of
            # being Pareto optimal, for efficiency
            prune_baseline=True,
            sampler=self.sampler,
        )
        # optimize
        new_x, _ = optimize_acqf(
            acq_function=acq_func,
            bounds=bounds,
            q=q_batch_size,
            num_restarts=self.num_restarts,
            raw_samples=self.raw_samples,  # used for intialization heuristic
            options={"batch_limit": self.batch_limit, "maxiter": self.maxiter},
            sequential=True,
        )
        new_x = new_x.detach()
        new_x = normalize(new_x, bounds)
        return {'new_x': new_x}


