from tqdm import tqdm
import torch
from botorch.utils.multi_objective.pareto import is_non_dominated
from botorch.acquisition.objective import GenericMCObjective
from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
from botorch.utils.sampling import sample_simplex
import logging
logger = logging.getLogger(__name__)


def eval_qnparego(f_cand, f_known):
    """Instantiate qNParEGO

    Parameters
    ----------
    f_cand : torch.Tensor
        Query candidates, of shape `[n_mc_samples, n_candidates, 1, f_dim]`
    f_known : torch.tensor
        Known f of shape [n_mc_samples, n_known, f_dim]

    Returns
    -------
    torch.tensor
        Acquisition values for the query candidates, of shape `[n_candidates,]`

    """
    num_objectives = f_cand.shape[-1]
    f_known_mean_pred = f_known.mean(0)  # [num_known, f_dim]
    weights = sample_simplex(
        num_objectives).squeeze().to(
            device=f_cand.device, dtype=f_cand.dtype)  # [num_objectives]
    objective_obj = GenericMCObjective(
        get_chebyshev_scalarization(
            weights=weights, Y=f_known_mean_pred))
    cand_obj = objective_obj(f_cand.squeeze(-2))  # [n_mc_samples, n_candidates]
    known_obj = objective_obj(f_known)  # [n_mc_samples, n_known]
    diff = cand_obj - known_obj.max(dim=-1, keepdim=True).values  # [n_mc_samples, n_candidates]
    ei = diff.clamp_min(0).mean(dim=0)  # [n_candidates,]
    return ei


def select_qnparego(y_pool, y_known, q_batch_size, ref_point, model, x_pool,
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
    for step_idx in tqdm(range(q_batch_size), desc='qnparego'):
        # Eval qNParEGO with new random weights for each q
        scores = eval_qnparego(y_pool, y_baseline_nd)  # [n_candidates,]
        # Compute pointwise scores, shape
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

    return dict(selected_idx=selected_idx,
                initial_nd=y_known_nd,
                final_nd=y_baseline_nd,
                scores_selected=scores_selected,)
