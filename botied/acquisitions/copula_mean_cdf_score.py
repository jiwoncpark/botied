from tqdm import tqdm
# import os
# os.unlink('gcc')
# os.unlink('g++')
# os.symlink('/home/ec2-user/miniconda3/envs/nard/bin/x86_64-conda-linux-gnu-cc', 'gcc')
# os.symlink('/home/ec2-user/miniconda3/envs/nard/bin/x86_64-conda-linux-gnu-cpp', 'g++')
import pyvinecopulib as pv
# from copulas.multivariate.gaussian import GaussianMultivariate
import numpy as np
# from scipy.spatial.distance import directed_hausdorff
import logging
logger = logging.getLogger(__name__)


# def instantiate_copula_v2(x):
#     cop = GaussianMultivariate()
#     cop.fit(x)
#     return cop


def instantiate_copula(u, weights=None, alpha=0.5, eps=1e-2):
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
        weights = np.ones(u.shape[0])
    weights = weights.reshape(-1, 1)
    # controls = pv.FitControlsVinecop(family_set=[pv.BicopFamily.tll] )#pv.BicopFamily.tll for nonparametric
    controls = pv.FitControlsVinecop(
        family_set=[pv.BicopFamily.gaussian])
        #family_set=[pv.BicopFamily.tll],
        # nonparametric_mult=0.001,
        #weights=weights)  # pv.BicopFamily.tll for nonparametric

    cop = pv.Vinecop(u, controls=controls)
    # dLc = cop.cdf(u)

    # level_line_idx = np.where(np.logical_and(dLc > alpha - eps, dLc <= alpha + eps))
    # d = x.shape[1]
    # dLF = [np.quantile(x[:, j], u[level_line_idx, j], method='inverted_cdf') for j in np.arange(d)]
    # dLF = np.asarray(dLF).squeeze().transpose()
    return cop  # , dLc , dLF


def select_copula_mean_cdf_score(
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
        Known f of shape [num_mc_samples, num_known, f_dim]
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
    # y_known_mean = y_known.mean(0)
    _, n_candidates, _ = y_pool.shape
    _, n_known, _ = y_known.shape
    selected_idx = []
    scores_selected = []
    y_selected = []
    # Select candidates sequentially

    for step_idx in tqdm(range(q_batch_size), desc='copula'):
        # Baseline
        # u_baseline = pv.to_pseudo_obs(y_known_mean)
        # baseline_copula = instantiate_copula(u_baseline, y_known_mean)
        # y_marginal = np.concatenate([y_pool.mean(0), y_known], axis=0)
        # y_std = np.concatenate(
        #     [y_pool.std(0), np.ones_like(y_known) * (y_pool.std(0).min())], axis=0)
        # u_candidate = pv.to_pseudo_obs(y_marginal)  # [n_cand+n_known, f_dim]
        '''
        # Candidate
        candidate_copula, dLc, dLF = get_pareto_front(
            u_candidate, y_marginal, weights=1.0/y_std)  # TODO: can use log likelihood ~ in-distribution score
        scores = np.empty(n_candidates)
        for cand_idx in range(n_candidates):
            #print(candidate_copula.cdf(u_candidate[cand_idx].reshape(-1, 2)))
            distance, _, _ = directed_hausdorff(
                u=baseline_copula.cdf(u_baseline).reshape(-1, 1),
                v=candidate_copula.cdf(u_candidate[:cand_idx].reshape(-1, 2)).reshape(-1, 1))

            #distance = baseline_copula.cdf(u_candidate)
            scores[cand_idx] = distance
        '''
        # option 2, scores same as joint cdf, higher score, closer to max pareto front
        # scores_m = baseline_copula.cdf(u_candidate[:-y_known.shape[0]].reshape(-1, 2))
        # y_pool_exp = y_pool.reshape(-1, 2)

        '''
        n_samples, _, f_dim = y_pool.shape
        _, n_known_so_far, _ = y_known.shape
        y_sample = np.concatenate([
            y_pool.reshape(-1, f_dim),  # 1 MC sample ~ [n_candidates*n_samples, f_dim]
            y_known.reshape(-1, f_dim)],  # 1 MC sample ~ [n_known_so_far*n_samples, f_dim]
            axis=0)  # [(n_candidate+n_known_so_far)*n_samples, f_dim]
        u_sample = pv.to_pseudo_obs(y_sample)
        copula = instantiate_copula(u_sample, weights=None)
        scores_samples = copula.cdf(
            u_sample,  # [(n_candidates+n_known_so_far)*n_samples, f_dim]
            N=2000)
        scores_samples = scores_samples.reshape(
            n_candidates+n_known_so_far, n_samples)
        scores = scores_samples.mean(1)[:n_candidates]
        '''
        n_samples, _, f_dim = y_pool.shape
        _, n_known_so_far, _ = y_known.shape
        u_baseline = pv.to_pseudo_obs(y_known.reshape(-1, f_dim))

        baseline_copula = instantiate_copula(u_baseline, y_known)
        y_marginal = np.concatenate([y_pool.mean(0).reshape(-1, f_dim), y_known.reshape(-1, f_dim)], axis=0)
        tmp = y_pool.mean(0).reshape(-1, f_dim)
        u_candidate = pv.to_pseudo_obs(y_marginal)  # [n_cand+n_known, f_dim]
        scores = baseline_copula.cdf(u_candidate[:tmp.shape[0]].reshape(-1, f_dim))

        # y_sample = np.concatenate([
        #     y_pool.reshape(-1, f_dim),  # 1 MC sample ~ [n_candidates*n_samples, f_dim]
        #     y_known.reshape(-1, f_dim)],  # 1 MC sample ~ [n_known_so_far*n_samples, f_dim]
        #     axis=0)  # [(n_candidate+n_known_so_far)*n_samples, f_dim]
        # sigma_sample = np.concatenate([
        #     y_pool.std(0),  # 1 MC sample ~ [n_candidates, f_dim]
        #     y_known.std(0)],  # 1 MC sample ~ [n_known_so_far, f_dim]
        #     axis=0)  # [(n_candidate+n_known_so_far), f_dim]
        # sigma_sample = np.tile(sigma_sample[np.newaxis, ...], reps=[n_samples, 1, 1])
        # weights = 1.0/np.sum(sigma_sample**2.0, -1)**0.5  # quadrature ~ [n_samples, n_candidate+n_known_so_far]
        # u_sample = pv.to_pseudo_obs(y_sample)
        # copula = instantiate_copula(u_sample, weights=weights.reshape(-1))
        # scores_samples = copula.cdf(
        #     u_sample,  # [(n_candidates+n_known_so_far)*n_samples, f_dim]
        #     N=2000)
        # scores_samples = scores_samples.reshape(
        #     n_candidates+n_known_so_far, n_samples)
        # scores = scores_samples.mean(1)[:n_candidates]
        # y_to_fit = np.concatenate([
        #     y_pool.mean(0),  # 1 MC sample ~ [n_candidates*n_samples, f_dim]
        #     y_known.mean(0)],  # 1 MC sample ~ [n_known_so_far*n_samples, f_dim]
        #     axis=0)  # [(n_candidate+n_known_so_far)*n_samples, f_dim]
        # y_to_eval = np.concatenate([
        #     y_pool.reshape(-1, f_dim),  # 1 MC sample ~ [n_candidates*n_samples, f_dim]
        #     y_known.reshape(-1, f_dim)],  # 1 MC sample ~ [n_known_so_far*n_samples, f_dim]
        #     axis=0)  # [(n_candidate+n_known_so_far)*n_samples, f_dim]
        # u_sample = pv.to_pseudo_obs(y_to_fit)
        # copula = instantiate_copula(u_sample)
        # u_to_eval = pv.to_pseudo_obs(y_to_eval)
        # scores_samples = copula.cdf(
        #     u_to_eval,  # [(n_candidates+n_known_so_far)*n_samples, f_dim]
        #     N=2000)
        # scores_samples = scores_samples.reshape(
        #     n_candidates+n_known_so_far, n_samples)
        # scores = scores_samples.mean(1)[:n_candidates]

        # for d in np.arange(n_samples):
            # Copula on a single sample of baseline points + candidates
            # y_sample = np.concatenate([
            #     y_pool[d, :, :],  # 1 MC sample ~ [n_candidates, f_dim]
            #     y_known[d, :, :]],  # 1 MC sample ~ [n_known, f_dim]
            #     axis=0)  # [n_candidates+n_known, f_dim]
            # Compute pseudo/ranking of each sample
            # u_sample = pv.to_pseudo_obs(y_sample)
            # copula = instantiate_copula(u_sample)
            # scores_candidates = copula.cdf(
            #     u_sample[:n_candidates, :],  # [n_candidates, f_dim]
            #     N=2000)
            # best_score_baseline = copula.cdf(
            #     u_sample[n_candidates:, :],  # [n_known, f_dim]
            #     N=1000).max(dim=0, keepdim=True).value  # [1, f_dim]
            # scores_samples[:, d] = scores_candidates   # - best_score_baseline
            # Using "Copulas" package with Gaussian assumption
            # baseline_copula = instantiate_copula_v2(y_known[d, :, :])
            # scores_samples[c, d] = baseline_copula.cumulative_distribution(
            #     y_pool[d, [c], :])  # scalar
            # u_candidate_sample = pv.to_pseudo_obs(y_marginal_sample).reshape(-1, f_dim)
        # final score will be mean + std??; mean * std
        # scores = scores_m + (scores_samples.std(1))
        # the one below makes more sense but the results are quite bad; no HV imporvement
        #scores = (scores_samples.mean(1))
        scores[selected_idx] = -1.0  # don't choose already chosen
        max_idx = scores.argmax().item()  # int scalar
        # Append selected candidate to solution
        selected_idx.append(max_idx)
        scores_selected.append(scores[max_idx].item())
        latest_selected_f = y_pool[:, max_idx, :]  # [n_mc_samples, f_dim]
        y_selected.append(latest_selected_f)
        # Append selected point to baseline
        y_known = np.concatenate([y_known, y_pool[:, [max_idx], :]], axis=1)
    # selected_idx ~ [q_batch_size,]
    y_selected = np.stack(y_selected, axis=1)  # [n_mc_samples, n_selected, f_dim]
    # Compute the joint batch score at the end
    if compute_joint_score:
        pass

    return dict(selected_idx=selected_idx,
                # joint_score=joint_score,
                # initial_nd=y_known_nd,
                # final_nd=y_baseline_nd,
                scores_selected=scores_selected, )
