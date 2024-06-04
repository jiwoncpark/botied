import torch
from botorch.utils.multi_objective.pareto import is_non_dominated
from copulala.acquisitions.qnehvi import get_joint_score
import logging
logger = logging.getLogger(__name__)


def select_random(
    y_pool, y_known, q_batch_size, ref_point, model, x_pool,
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
        ref_point : torch.Tensor
            Unused  # FIXME jwp
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
    # Select random (first q_batch_size candidates is fine)
    y_selected = y_pool[:, :q_batch_size, :]  # [num_mc_samples, q_batch_size, f_dim]
    # Get final known_nd
    # Update baseline set to include posterior mean of selected candidate
    y_baseline_nd = torch.cat(
        [
            y_known_nd.clone(),  # [n_baseline_samples, num_nd, f_dim]
            # Take arbitrary subset from available posterior samples
            y_selected[:n_baseline_samples, :],
            # ~ [n_baseline_samples, q_batch_size, f_dim]
        ],
        dim=1
    )  # [n_baseline_samples, num_known+(step_idx+1), f_dim]
    is_non_dom = is_non_dominated(y_baseline_nd.mean(0))  # based on posterior mean
    y_baseline_nd = y_baseline_nd[:, is_non_dom, :]
    # Compute the joint batch score at the end
    if compute_joint_score:
        joint_score = get_joint_score(y_known_nd, y_selected, ref_point)
    else:
        joint_score = None
    return dict(
        selected_idx=list(range(q_batch_size)),
        joint_score=joint_score,
        initial_nd=y_known_nd,
        final_nd=y_baseline_nd,
        scores_selected=joint_score,)
