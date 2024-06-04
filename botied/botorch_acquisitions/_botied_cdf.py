from typing import Optional
import torch
from botorch.acquisition.monte_carlo import MCAcquisitionFunction
from botorch.models.model import Model
from botorch.sampling.base import MCSampler
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.utils import t_batch_mode_transform
from torch import Tensor
from botied.cdf import CDFModel
import logging
logger = logging.getLogger(__name__)


class BOtiedCDF(MCAcquisitionFunction):
    def __init__(
        self,
        model: Model,
        observed_x: Tensor,
        sampler: Optional[MCSampler] = None,
        num_posterior_samples: int = 128,
        cdf_model: CDFModel = CDFModel(),
        aggregation: str = 'mean_of_cdfs',
        ref_point: Optional[Tensor] = None,
        X_pending: Optional[Tensor] = None,
    ) -> None:
        """

        Parameters
        ----------
        observed_x : Tensor
            All the inputs so far for which we have function evaluations

        """
        # we use the AcquisitionFunction constructor, since that of
        # MCAcquisitionFunction performs some validity checks that we don't want here
        super(MCAcquisitionFunction, self).__init__(model=model)
        self.num_posterior_samples = num_posterior_samples
        if sampler is None:
            sampler = SobolQMCNormalSampler(
                sample_shape=torch.Size([self.num_posterior_samples]))
        self.sampler = sampler
        self.num_objectives = self.model._num_outputs
        self.cdf_model = cdf_model
        self.aggregation = aggregation
        self.ref_point = ref_point
        if ref_point is not None:
            self.ref_point = ref_point.detach().cpu().numpy()  # [M,]
        # Cache for copula fitting
        self.observed_x = observed_x.unsqueeze(1)  # [num_instances, 1, M]
        self.set_X_pending(X_pending)

    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        """Evaluate scalarized qUCB on the candidate set `X`.

        Args:
            X: A `(b) x q x d`-dim Tensor of `(b)` t-batches with `q` `d`-dim
                design points each.

        Returns:
            Tensor: A `(b)`-dim Tensor of Upper Confidence Bound values at the
                given design points `X`.
        """
        if X.shape[-2] != 1:
            raise ValueError("`q` must be 1")
        num_candidates = X.shape[0]
        num_baseline = self.observed_x.shape[0]
        num_total = num_candidates + num_baseline
        total_x = torch.cat([X, self.observed_x], dim=0)
        # dim 0 becomes num_candidates + num_baseline
        posterior = self.model.posterior(total_x)
        # Scoring (CDF fitting and evaluation)
        # TODO jwp: just use aggregation to getattr method
        if self.aggregation == "mean_of_cdfs":  # v1
            scores = self.get_scores_for_mean_of_cdfs(posterior)
        elif self.aggregation == "cdf_of_means":  # v2
            scores = self.get_scores_for_cdf_of_means(posterior)
        elif self.aggregation == "cdf_of_means_eval_samples":  # v3
            scores = self.get_scores_for_cdf_of_means_eval_samples(posterior)
        else:
            raise ValueError("Invalid aggregation.")
        scores = scores[:num_candidates]
        scores = torch.from_numpy(scores)
        return scores

    def get_scores_for_cdf_of_means(self, posterior):
        f_means = posterior.mean  # [num_total, q=1, M]
        f_means = f_means.detach().cpu().numpy()
        f_means = f_means.squeeze(-2)  # ~ [num_total, M]
        scores = self.cdf_model(f_means)  # [num_total,]
        if self.ref_point is not None:
            dom_by_ref = (f_means < self.ref_point).any(1)
            scores[dom_by_ref] = 0.0
        return scores

    def get_scores_for_mean_of_cdfs(self, posterior):
        f_samples = self.get_posterior_samples(posterior)
        f_samples = f_samples.detach().cpu().numpy()
        num_total = f_samples.shape[1]
        # ~ [num_samples, num_total, q=1, M]
        f_samples = f_samples.squeeze(-2).reshape(-1, self.num_objectives)
        # ~ [num_samples*num_total, M]
        scores = self.cdf_model(f_samples)  # ~ [num_samples*num_total,]
        if self.ref_point is not None:
            dom_by_ref = (f_samples < self.ref_point).any(1)
            scores[dom_by_ref] = 0.0
        scores = scores.reshape(-1, num_total).mean(0)  # ~ [num_total,]
        return scores

    def get_scores_for_cdf_of_means_eval_samples(self, posterior):
        # Means
        f_means = posterior.mean  # [num_total, q=1, M]
        f_means = f_means.detach().cpu().numpy()
        f_means = f_means.squeeze(-2)  # ~ [num_total, M]
        # Samples
        f_samples = self.get_posterior_samples(posterior)
        f_samples = f_samples.detach().cpu().numpy()
        num_total = f_samples.shape[1]
        # ~ [num_samples, num_total, q=1, M]
        f_samples = f_samples.squeeze(-2).reshape(-1, self.num_objectives)
        # ~ [num_samples*num_total, M]
        scores = self.cdf_model(
            f_means, eval_y=f_samples)  # [num_samples*num_total,]
        if self.ref_point is not None:
            dom_by_ref = (f_samples < self.ref_point).any(1)
            scores[dom_by_ref] = 0.0
        scores = scores.reshape(-1, num_total).mean(0)  # ~ [num_total,]
        return scores


if __name__ == "__main__":
    import torch

    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from botorch.utils import standardize
    from gpytorch.mlls import ExactMarginalLogLikelihood


    # generate synthetic data
    X = torch.rand(7, 2)
    Y = torch.stack([torch.sin(X[:, 0]), torch.cos(X[:, 1])], -1)  # [7, 2]
    Y = standardize(Y)  # standardize to zero mean unit variance
    observed_x = torch.rand(20, 2)

    # construct and fit the multi-output model
    gp = SingleTaskGP(X, Y)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)

    # construct the acquisition function
    botied_copula_cdf = BOtiedCDF(
        gp, observed_x=observed_x)
    scores = botied_copula_cdf(torch.rand(8, 1, 2))
    print(scores.shape)
