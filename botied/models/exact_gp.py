from typing import Union
import torch
import gpytorch
from botorch.models.transforms import Standardize, Normalize
from botorch.models.gp_regression import SingleTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import (
    ExactMarginalLogLikelihood)
from botorch.fit import fit_gpytorch_mll
from gauche.kernels.fingerprint_kernels.tanimoto_kernel import TanimotoKernel
from botied.models.base_model import BaseModel


# class ExactGPModel(gpytorch.models.ExactGP):
#     def __init__(self, train_x, train_y, likelihood):
#         super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
#         self.mean_module = gpytorch.means.ConstantMean()
#         self.covar_module = gpytorch.kernels.ScaleKernel(
#             gpytorch.kernels.RBFKernel())

#     def forward(self, x):
#         mean_x = self.mean_module(x)
#         covar_x = self.covar_module(x)
#         return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class ExactGP(BaseModel):
    """
    Convenience class that manages model fitting
    Always standardizes output
    Input must be separately normalized/standardized

    """
    def __init__(self, kernel: Union[str, None] = None):
        super(ExactGP, self).__init__()
        if kernel is None:
            self.covar_module = None
        elif kernel.lower() == 'tanimoto':
            self.covar_module = gpytorch.kernels.ScaleKernel(TanimotoKernel())
        else:
            raise ValueError("Kernel not supported")  # FIXME jwp

    def fit_model(self, train_x, train_obj, bounds):
        """Fit a SingleTaskGP model and associated marginal log likelihood
        object

        Parameters
        ----------
        train_x : torch.Tensor
            Unused
        train_obj : torch.Tensor
            Unused

        Returns
        -------
        Tuple (model, likelihood

        """
        # if self.standardize_x:
        #     self.x_bounds = torch.stack([train_x.min(0)[0], train_x.max(0)[0]], dim=0)
        #     train_x = normalize(train_x, self.x_bounds)
        # if self.standardize_obj:
        #     self.obj_bounds = torch.stack([train_obj.min(0)[0], train_obj.max(0)[0]], dim=0)
        #     train_obj = normalize(train_obj, self.obj_bounds)
        model = SingleTaskGP(
            train_x, train_obj,
            covar_module=self.covar_module,
            # outcome_transform=Standardize(m=train_obj.shape[-1]),
            # input_transform=Normalize(
            #     d=train_x.shape[-1],
            #     # bounds=bounds),
        ).to(device=self.device, dtype=self.dtype)
        model.train()
        model.likelihood.train()
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        mll = fit_gpytorch_mll(mll)
        return model, model.likelihood

    def sample_predictions(self, model, likelihood, input_x, n_samples):
        """
        Sample posterior predictions from a fit model

        Returns
        -------
        torch.Tensor
            Predictive samples of shape `[n_samples, len(test_x), y_dim]`

        """
        sample_size = torch.Size([n_samples])
        model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            # predictions = likelihood(model(test_x))
            # if self.standardize_x:
            #     input_x = normalize(model_input['x'], self.x_bounds)
            # transformed_input_x = model.input_transform(input_x)
            samples = model.posterior(
                input_x).sample(sample_size)  # [n_samples, len(test_x), y_dim]
            # if self.standardize_obj:
            #     samples = unnormalize(samples, self.obj_bounds)
        return samples
