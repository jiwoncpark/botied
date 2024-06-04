import pyvinecopulib as pv
import numpy as np
import torch
from torch import Tensor
from matplotlib import pyplot as plt
from botied.cdf import CDFModel
# from botorch.utils.multi_objective.box_decompositions.dominated import (
#     DominatedPartitioning)
import logging
logger = logging.getLogger(__name__)


class CDFIndicator:
    def __init__(self, cdf_model: CDFModel = CDFModel()):
        """
        Parameters
        ----------
        num_mc_samples : int
            Integer for the number of quasi-random numbers to draw to evaluate
            the distribution.

        """
        self.cdf_model = cdf_model
        self.data_list = []  # data['y'].detach().cpu().numpy()

    def compute_lookback_copula_cdf(self, evaluated_points: dict) -> dict:
        """
        Retrospectively compute CDF score per iter after the final iter

        Parameters
        ----------
        evaluated_points : dict
            Each key is an acq function name, each value is a list of candidate
            outcomes tensors, each element representing an iteration

        Returns
        -------
        dict
            List of CDF scores across iterations, for each acq function

        """
        acq_names = list(evaluated_points.keys())  # A = len(acq_names)
        num_per_acq = len(
            np.concatenate(evaluated_points[acq_names[0]][1:], axis=0))  # num_eval
        init_y = evaluated_points[acq_names[0]][0]
        # ~ [num_init,] same for all acq
        num_init = init_y.shape[0]
        all_y = np.concatenate(
            [init_y] +
            [np.concatenate(evaluated_points[acq_name][1:], axis=0) for acq_name in acq_names],
            axis=0
        )  # [num_init + A*num_eval, M]
        all_cdf_scores = self.cdf_model(all_y)
        # all_u_samples = pv.to_pseudo_obs(
        #     all_y)  # [num_init + A*num_eval, M]
        # copula = pv.Vinecop(all_u_samples, controls=self.controls)
        # all_cdf_scores = copula.cdf(
        #     all_u_samples,
        #     N=self.num_mc_samples)  # [num_init + A*num_eval,]
        init_cdf_score = all_cdf_scores[:num_init].max()
        batch_cdf_scores = all_cdf_scores[num_init:]
        # Get CDF scores for initial iteration (iter 0)
        lookback_cdf_scores_dict = {}
        for acq_i, acq_name in enumerate(acq_names):
            y_batch_list = evaluated_points[acq_name][1:]  # exclude init
            cdf_scores_acq = batch_cdf_scores[
                acq_i*num_per_acq: (acq_i+1)*num_per_acq]  # [num_eval,]
            lookback_cdf_scores = [init_cdf_score]
            num_y_so_far = num_init
            for iter_i, y_batch in enumerate(y_batch_list):
                num_y_so_far += len(y_batch)
                cdf_score = cdf_scores_acq[:num_y_so_far]
                cdf_score_iter = cdf_score.max()
                lookback_cdf_scores.append(cdf_score_iter)
            lookback_cdf_scores_dict[acq_name] = lookback_cdf_scores
        return lookback_cdf_scores_dict

    # def slice_into_iterations(self, candidate_Y_list: list) -> list:
    #     """
    #     Retrospectively compute CDF score per iter after the final iter

    #     Parameters
    #     ----------
    #     candidate_Y_list : list
    #         List of candidate outcomes tensors, each element representing an
    #         iteration

    #     Returns
    #     -------
    #     list
    #         List of CDF scores across iterations

    #     """
    #     all_y = np.concatenate(
    #         candidate_Y_list,
    #         axis=0)
    #     all_u_samples = pv.to_pseudo_obs(all_y)
    #     copula = pv.Vinecop(all_u_samples, controls=self.controls)
    #     all_cdf_scores = copula.cdf(all_u_samples, N=self.num_mc_samples)
    #     num_y_so_far = 0
    #     retro_cdf_scores = []
    #     for iter_i, candidate_Y in enumerate(candidate_Y_list):
    #         num_y_so_far += len(candidate_Y)
    #         cdf_score = all_cdf_scores[:num_y_so_far]
    #         cdf_score_iter = cdf_score.max()
    #         retro_cdf_scores.append(cdf_score_iter)
    #     return retro_cdf_scores

    def compute_copula_cdf(self, candidate_Y: Tensor) -> float:
        """Compute cdf score.
            Args:
                candidate_pareto_Y: A `n x m`-dim tensor of candidate outcomes
                CDF is fit on all the observed Y data so far

            Returns:
                The evaluated cdf score on the new candidate_Y.
        """
        candidate_Y = candidate_Y.detach().cpu().numpy()
        num_candidates = candidate_Y.shape[0]
        self.data_list.append(candidate_Y)
        y_sample = np.concatenate(
            self.data_list,
            axis=0)
        cdf_score = self.cdf_model(y_sample)
        return cdf_score.max()
        # cdf_score = self.copula.cdf(u_sample[:, :], N=10000)
        # self.track_data = np.concatenate(
        #     [candidate_Y, self.track_data],
        #     axis=0)

        if False:
            u_sim = self.copula.simulate(100)
            sel = 5
            idx = (-cdf_score).argsort()[:sel]
            print(cdf_score[idx])
            plt.scatter(y_sample[:, 0], y_sample[:, 1], c=np.array(cdf_score).reshape(-1), cmap="plasma", alpha=0.5)
            plt.colorbar()

            plt.scatter(y_sample[idx, 0], y_sample[idx, 1], c='grey',  facecolors='none',alpha=0.5, s=60)

            plt.title('CDF metric')
            plt.show()

            from sklearn.preprocessing import MinMaxScaler
            min_max_scaler = MinMaxScaler()
            y_sample_sc = min_max_scaler.fit_transform(y_sample)
            u_sample_sc = pv.to_pseudo_obs(y_sample_sc)

            plt.plot()

            cdf_score_sc = self.copula.cdf(u_sample[:, :], N=10000)
            idx_sc = (-cdf_score_sc).argsort()[:sel]
            print(idx_sc)
            print(cdf_score_sc[idx_sc])
            plt.scatter(y_sample[:, 0], y_sample[:, 1], c=np.array(cdf_score_sc).reshape(-1), cmap="plasma", alpha=0.5)
            plt.colorbar()
            plt.scatter(y_sample[idx_sc, 0], y_sample[idx_sc, 1], c='grey',facecolors='none', alpha=0.5, s=60)

            plt.title('CDF indicator after scaling')


            plt.show()

            hvs = []
            ref_point = torch.Tensor(np.array([-1.0, -1.0]))
            for i in np.arange(y_sample.shape[0]):
                y_sample_2 = torch.Tensor(y_sample[i].reshape(-1, 2))
                bd = DominatedPartitioning(
                    ref_point=ref_point,
                    Y=y_sample_2  # observed (noisy) y
                )
                hvs.append(bd.compute_hypervolume().item())
            hvs = np.array(hvs)
            idx = (-hvs).argsort()[:sel]
            print(hvs[idx])
            plt.scatter(y_sample[:, 0], y_sample[:, 1], c=hvs, cmap="plasma", alpha=0.5)
            plt.colorbar()
            plt.scatter(y_sample[idx, 0], y_sample[idx, 1],  c='grey', facecolors='none', alpha=0.5, s=60)

            plt.title('HV indicator')

            plt.show()

            from sklearn.preprocessing import MinMaxScaler
            min_max_scaler = MinMaxScaler()
            y_sample_sc = min_max_scaler.fit_transform(y_sample)

            hvs_sc = []
            ref_point = torch.Tensor(np.array([-1.0, -1.0]))
            for i in np.arange(y_sample.shape[0]):
                y_sample_2 = torch.Tensor(y_sample_sc[i].reshape(-1, 2))
                bd = DominatedPartitioning(
                    ref_point=ref_point,
                    Y=y_sample_2  # observed (noisy) y
                )
                hvs_sc.append(bd.compute_hypervolume().item())
            hvs_sc = np.array(hvs_sc)
            idx_sc = (-hvs_sc).argsort()[:sel]
            print(hvs_sc[idx_sc])
            plt.scatter(y_sample[:, 0], y_sample[:, 1], c=hvs_sc, cmap="plasma", alpha=0.5)
            plt.colorbar()
            plt.scatter(y_sample[idx_sc, 0], y_sample[idx_sc, 1],  c='grey', facecolors='none',alpha=0.5, s=60)

            plt.title('HV indicator after scaling')

            plt.show()
        #return 1-(cdf_score.max())/cdf_score.sum()
