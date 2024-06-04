import numpy as np
import cma
import torch
from botorch.sampling.normal import SobolQMCNormalSampler
from copulala.botorch_acquisitions import BOtiedCDF as BoTorchBOtiedCDF
from .base_acquisition import BaseAcquisition
from copulala.cdf import CDFModel
# from botorch.utils.transforms import unnormalize, normalize
import logging
logger = logging.getLogger(__name__)


class BOtiedCDF(BaseAcquisition):

    def __init__(
            self,
            cdf_model: CDFModel = CDFModel(),
            aggregation: str = 'mean_of_cdfs',
            apply_ref_point: bool = False,
            num_posterior_samples: int = 64,
            popsize: int = 1024,
            maxiter: int = 32,
            sigma0: float = 0.2):
        super().__init__()
        self.cdf_model = cdf_model
        self.aggregation = aggregation
        self.popsize = popsize
        self.maxiter = maxiter
        self.sigma0 = sigma0
        self.num_posterior_samples = num_posterior_samples
        self.apply_ref_point = apply_ref_point
        if self.aggregation == 'mean_of_cdfs':
            # Reduce pop size by num_posterior_samples
            # because effective pop size is
            # original popsize x num_posterior_samples
            self.popsize = self.popsize // self.num_posterior_samples
        self.sampler = SobolQMCNormalSampler(
            sample_shape=torch.Size([self.num_posterior_samples]))
        logger.info(f"popsize: {self.popsize}")
        logger.info(f"maxiter: {self.maxiter}")
        logger.info(f"sigma0: {self.sigma0}")

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
            Baseline f of shape [n_mc_samples, num_baseline, f_dim]

        Returns
        -------
        torch.Tensor
            scores

        """
        num_candidates = f_cand.shape[1]
        num_baseline = f_baseline.shape[1]
        num_objectives = f_cand.shape[-1]
        num_total = num_candidates + num_baseline

        # CDF fitting and evaluation with pyvinecopulib
        f_samples = torch.cat(
            [f_cand, f_baseline], dim=1
        )  # [N, B = B1 + B2, M]
        f_samples = f_samples.detach().cpu().numpy()  # [N, B, M]
        if self.aggregation == 'cdf_of_means':
            f_samples = f_samples.mean(0)  # [B, M]
        f_samples = f_samples.reshape(-1, num_objectives)
        # mean_of_cdfs: [N*B, M]
        # cdf_of_means: [B, M]
        scores = self.cdf_model(f_samples)
        # mean_of_cdfs: [N*B,]
        # cdf_of_means: [B,]
        if self.aggregation == 'mean_of_cdfs':
            scores = scores.reshape(-1, num_total).mean(0)  # [B,]
        # scores ~ [B,]
        scores = scores[:num_candidates]  # [B1=num_candidates,]
        scores = torch.from_numpy(scores)
        return scores

    def optimize(
            self, model, train_x, bounds, q_batch_size, ref_point,
            *args, **kwargs):
        """"""
        acq_function_list = [BoTorchBOtiedCDF(
            model,
            observed_x=train_x,
            sampler=self.sampler,
            cdf_model=self.cdf_model,
            aggregation=self.aggregation,
            ref_point=ref_point if self.apply_ref_point else None,
        ) for _ in range(q_batch_size)]
        base_X_pending = acq_function_list[0].X_pending

        # Create the CMA-ES optimizer
        # bounds_cpu = bounds.detach().cpu()
        # minmax_bounds = [
        #         bounds_cpu[0].min().item(),
        #         bounds_cpu[1].max().item()]  # min, max across all f_dim
        minmax_bounds = [0.0, 1.0]  # FIXME assumes input always 0, 1 normed
        # x0 = torch.rand(
        #     model._num_outputs)*(bounds_cpu[1] - bounds_cpu[0]) + bounds_cpu[0]
        # x0 = x0.numpy()
        # Enter sequential greedy loop. See:
        # https://botorch.org/api/_modules/botorch/optim/optimize.html#optimize_acqf_list
        candidate_list = []
        candidates = torch.tensor([], device=bounds.device, dtype=bounds.dtype)
        for batch_idx in range(q_batch_size):
            # Speed up by telling pytorch not to generate a compute graph
            with torch.no_grad():
                botorch_acq = acq_function_list[batch_idx]
                botorch_acq.set_X_pending(
                    torch.cat([base_X_pending, candidates], dim=-2)
                    if base_X_pending is not None else candidates
                )
                # Run the optimization loop using the ask/tell interface
                # This uses PyCMA's default settings
                # Do not use Bessel's correction (default for numpy)
                init_data = torch.cat(
                    [train_x, candidates], dim=0).cpu().numpy()  # [N, d]
                x0 = init_data[-2]
                alpha = 0.8
                x0 = np.random.rand(init_data.shape[-1])*alpha + x0*(1.0 - alpha)
                # stds = init_data.std(0, ddof=0)
                # TODO jwp: init at the previous best or is cold start better?
                # TODO jwp: estimate reasonable per-objective sigma0 value
                es = cma.CMAEvolutionStrategy(
                    # x0=x0,
                    x0=x0,
                    sigma0=self.sigma0,
                    # tolupsigma=(init_data.shape[-1])**0.5,  # sqrt(d)
                    inopts={
                        "bounds": minmax_bounds,
                        "popsize": self.popsize})
                # List of options: https://github.com/CMA-ES/pycma/issues/171
                es.opts.set({
                    'maxiter': self.maxiter,
                    })
                logger.debug(f"Item {batch_idx} in batch")
                iter_i = 0
                while not es.stop():
                    xs = es.ask()  # ask for new points to evaluate
                    # xs ~ list of np array candidates
                    xs_stacked = np.stack(xs, axis=0)  # [popsize, x_dim]
                    # convert to Tensor for evaluating the acquisition function
                    X = torch.tensor(
                        xs_stacked, device=bounds.device, dtype=bounds.dtype)
                    # evaluate the acquisition function (optimizer assumes we're minimizing)
                    Y = -botorch_acq(
                        X.unsqueeze(-2)
                    )  # acquisition functions require an explicit q-batch dimension
                    y = Y.view(-1).double().numpy()  # convert result to numpy array
                    es.tell(xs, y)  # return the result to the optimizer
                    if iter_i == 0:
                        logger.debug(
                        f"Iteration {iter_i}: max acq value: {(-y).max()}")
                        first_iter_acq = (-y).max()
                    logger.debug(f"sigma, max acq: {es.sigma, (-y).max()}")
                    logger.debug(
                        f"Iteration {iter_i}: max acq value: {(-y).max()}")
                    iter_i += 1
                logger.debug(
                    f"Terminated in {iter_i} iterations,"
                    f" last acq value: {(-y).max()}"
                    f" improved since first iter: {-es.result.fbest - first_iter_acq}"
                    f" best acq value: {-es.result.fbest}"
                    )
            # convert result back to a torch tensor
            best_x = torch.from_numpy(es.best.x).to(
                device=X.device, dtype=X.dtype)
            candidate_list.append(best_x)
            candidates = torch.stack(candidate_list, dim=0)  # [batch_idx, num_objectives]
        return {"new_x": candidates}
