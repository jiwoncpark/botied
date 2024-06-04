from tqdm import tqdm
import torch
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.utils.multi_objective.box_decompositions \
    import NondominatedPartitioning
from botorch.acquisition.multi_objective.utils import (
    compute_sample_box_decomposition
)
from botorch.acquisition.multi_objective.joint_entropy_search import (
    qLowerBoundMultiObjectiveJointEntropySearch,
)
import logging
logger = logging.getLogger(__name__)


def jes_callback(f_cand, model, x_cand, *args, **kwargs):
    """Wrapper around `qExpectedHypervolumeImprovement` to access acquisition
    values directly from samples

    Parameters
    ----------
    f_non_dom : torch.tensor
        Known points that are non-dominated, of shape `[num_nd, f_dim]`
    f_cand : torch.tensor
        Query candidates from a sample path, of shape `[n_candidates, f_dim]`
    model : Model
        Fitted surrogate

    Returns
    -------
    torch.tensor
        Acquisition values for the query candidates, of shape `[n_candidates,]`

    """
    # Find non-dominated points among f_cand
    is_nd = is_non_dominated(f_cand)
    f_nd = f_cand[is_nd]
    x_nd = x_cand[is_nd]
    hypercell_bounds = compute_sample_box_decomposition(f_nd.unsqueeze(0))
    jes_lb_obj = qLowerBoundMultiObjectiveJointEntropySearch(
        model=model,
        pareto_sets=x_nd.unsqueeze(0),
        pareto_fronts=f_nd.unsqueeze(0),
        hypercell_bounds=hypercell_bounds,
        estimation_type="LB",
    )
    scores = jes_lb_obj(x_cand.unsqueeze(-2))
    return scores


def select_jes(y_pool, y_known, q_batch_size, ref_point, model, x_pool,
                  compute_joint_score=False):
    """Select a batch sequence by sequence, based on pre-fetched oracle
    samples
    jwp: Do not turn into classes yet -- some speed refactor pending

    Parameters
    ----------
        y_pool : torch.tensor
            Candidate f of shape [num_mc_samples, num_candidates, f_dim]
        y_known : torch.tensor
            Known f of shape [n_baseline_samples, num_known, f_dim]
        q_batch_size : int
            How many candidates to select

    Returns
    -------
    selected_idx : list
        Indices of selected candidates from `y_pool`, of length
        `q_batch_size`
    initial_nd : torch.Tensor
    final_nd : torch.Tensor
    scores_selected : list

    """
    if y_known.dim() == 2:
        y_known = y_known.unsqueeze(0)  # n_baseline_samples = 1
    n_mc_samples, n_candidates, f_dim = y_pool.shape
    n_baseline_samples, n_known, f_dim_baseline = y_known.shape
    if q_batch_size >= n_candidates:
        logger.info("Selecting all of candidate pool")
        q_batch_size = n_candidates
    if n_baseline_samples > n_mc_samples:  # FIXME jwp: arbitrary restriction
        raise ValueError(
            "Number of posterior samples for candidates must be greater than"
            "or equal to that for known points.")
    if f_dim != f_dim_baseline:
        raise ValueError(
            "Candidate and known objective dimensions must be the same.")

    # Initialize baseline points, reference point
    is_non_dom = is_non_dominated(y_known.mean(0))  # based on posterior mean
    y_known_nd = y_known[:, is_non_dom, :]
    y_baseline_nd = y_known_nd.clone()  # [n_baseline_samples, num_nd, f_dim]
    selected_idx, y_selected, scores_selected = [], [], []

    # Select candidates sequentially
    for step_idx in tqdm(range(q_batch_size), desc='jes'):
        # Compute pointwise scores, shape [n_candidates,]
        scores = 0.0
        # FIXME jwp: this is slow -- access botorch jes directly
        for sample_i in range(n_baseline_samples):
            baseline = y_baseline_nd[sample_i]  # [n_nd, f_dim]
            scores_i = jes_callback(
                f_cand=y_pool[sample_i],  # [n_candidates, f_dim]
                model=model,
                x_cand=x_pool,  # [n_candidates, d]
            )
            scores += scores_i  # TODO jwp: switch to running mean
        scores = scores/n_baseline_samples  # [n_candidates,]
        scores[selected_idx] = -1.0  # don't choose already chosen
        max_idx = scores.argmax().item()  # int scalar

        # Append selected candidate to solution
        selected_idx.append(max_idx)
        scores_selected.append(scores[max_idx].item())
        latest_selected_f = y_pool[:, max_idx, :]  # [n_mc_samples, f_dim]
        y_selected.append(latest_selected_f)

        # Update baseline set to include posterior mean of selected candidate
        y_baseline_nd = torch.cat(
            [
                y_baseline_nd,  # [n_baseline_samples, num_nd, f_dim]
                # Take arbitrary subset from available posterior samples
                latest_selected_f[:n_baseline_samples, :].unsqueeze(1),
                # ~ [n_baseline_samples, 1, f_dim]
            ],
            dim=1
        )  # [n_baseline_samples, num_known+(step_idx+1), f_dim]
        is_non_dom = is_non_dominated(y_baseline_nd.mean(0))  # based on posterior mean
        y_baseline_nd = y_baseline_nd[:, is_non_dom, :]

    # selected_idx ~ [q_batch_size,]
    y_selected = torch.stack(y_selected, dim=1)  # [n_mc_samples, n_selected, f_dim]
    # Compute the joint batch score at the end
    if compute_joint_score:
        raise NotImplementedError

    return dict(selected_idx=selected_idx,
                joint_score=None,
                initial_nd=y_known_nd,
                final_nd=y_baseline_nd,
                scores_selected=scores_selected,)

