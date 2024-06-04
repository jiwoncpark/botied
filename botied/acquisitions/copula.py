from tqdm import tqdm
import pyvinecopulib as pv
import numpy as np
from scipy.spatial.distance import directed_hausdorff
import logging
logger = logging.getLogger(__name__)


def get_pareto_front(u, x, weights=None, alpha=0.9999, eps=1e-3):
    """

    Parameters
    ----------
    u : np.ndarray
    x : np.ndarray
        In the data space
    weights : np.ndarray
    alpha : float
        Level line

    """
    # fit a copula to pseudo observations
    # evaluate joint cdf
    # get level line dL_\alpha^C
    # use inverse margins to get level line dL_\alpha^F
    if weights is None:
        weights = np.ones(x.shape[0])
    controls = pv.FitControlsVinecop(family_set=[pv.BicopFamily.tll], weights=weights) #pv.BicopFamily.tll for nonparametric
    cop = pv.Vinecop(u, controls=controls)
    dLc = cop.cdf(u)

    level_line_idx = np.where(np.logical_and(dLc > alpha-eps, dLc <= alpha+eps))
    d = x.shape[1]
    dLF = [np.quantile(x[:, j], u[level_line_idx, j], method='inverted_cdf') for j in np.arange(d)]
    dLF = np.asarray(dLF).squeeze().transpose()
    print(dLF.shape)
    return cop, dLc, dLF


def select_copula(y_pool, y_known, q_batch_size, ref_point,
                  compute_joint_score=True):
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
    y_pool = y_pool.cpu().numpy()
    y_known = y_known.cpu().numpy()
    _, n_candidates, _ = y_pool.shape
    selected_idx = []
    scores_selected = []
    y_selected = []
    # Select candidates sequentially
    for step_idx in tqdm(range(q_batch_size), desc='copula'):
        # Baseline
        u_baseline = pv.to_pseudo_obs(y_known)
        baseline_copula, _, _ = get_pareto_front(u_baseline, y_known)
        y_marginal = np.concatenate([y_pool.mean(0), y_known], axis=0)
        y_std = np.concatenate(
            [y_pool.std(0), np.ones_like(y_known)*(y_pool.std(0).min())], axis=0)
        u_candidate = pv.to_pseudo_obs(y_marginal)  # [n_cand+n_known, f_dim]

        # Baseline
        u_baseline = pv.to_pseudo_obs(y_known)
        baseline_copula, _, _ = get_pareto_front(u_baseline, y_known)
        # Candidate
        y_marginal = np.concatenate([y_pool.mean(0), y_known], axis=0)
        y_std = np.concatenate(
            [y_pool.std(0), np.ones_like(y_known)*(y_pool.std(0).min())], axis=0)
        u_candidate = pv.to_pseudo_obs(y_marginal)   # [n_cand+n_known, f_dim]
        candidate_copula, dLc, dLF = get_pareto_front(
            u_candidate, y_marginal, weights=1.0/y_std)  # TODO: can use log likelihood ~ in-distribution score
        scores = np.empty(n_candidates)
        for cand_idx in range(n_candidates):
            distance, _, _ = directed_hausdorff(
                u=baseline_copula.cdf(u_baseline),
                v=candidate_copula.cdf(u_candidate))
            scores[cand_idx] = distance
        scores[selected_idx] = -1.0  # don't choose already chosen
        max_idx = scores.argmax().item()  # int scalar

        # Append selected candidate to solution
        selected_idx.append(max_idx)
        scores_selected.append(scores[max_idx].item())
        latest_selected_f = y_pool[:, max_idx, :]  # [n_mc_samples, f_dim]
        y_selected.append(latest_selected_f)
    # selected_idx ~ [q_batch_size,]
    y_selected = np.stack(y_selected, axis=1)  # [n_mc_samples, n_selected, f_dim]
    # Compute the joint batch score at the end
    if compute_joint_score:
        pass

    return dict(selected_idx=selected_idx,
                # joint_score=joint_score,
                # initial_nd=y_known_nd,
                # final_nd=y_baseline_nd,
                scores_selected=scores_selected,)
