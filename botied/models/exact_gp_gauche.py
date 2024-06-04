import warnings
warnings.filterwarnings("ignore") # Turn off Graphein warnings
from botorch import fit_gpytorch_model
import gpytorch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import torch
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.exact_marginal_log_likelihood import (
    ExactMarginalLogLikelihood)
from botorch.fit import fit_gpytorch_mll
from gauche.dataloader import DataLoaderMP
from gauche.dataloader.data_utils import transform_data
from botied.models.base_model import BaseModel


class ExactGPTanimoto(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPTanimoto, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        # We use the Tanimoto kernel to work with molecular fingerprint representations
        self.covar_module = gpytorch.kernels.ScaleKernel(TanimotoKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class ExactGPGauche(BaseModel):
    def __init__(self):
        super(ExactGPGauche, self).__init__()
        self.loader = DataLoaderMP()

    def init_model(self, context):
        """Initialize a SingleTaskGP model and associated marginal log likelihood
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
        train_x = context['x']
        train_obj = context['y']
        num_outputs = train_obj.shape[-1]
        model = ExactGPTanimoto(
            train_x, train_obj,
            outcome_transform=Standardize(m=num_outputs)).to(
                device=self.device, dtype=self.dtype)
        return model, model.likelihood

    def fit_model(self, model, likelihood, context=None):
        """Fit a provided model

        Parameters
        ----------
        model : torch.nn.Module object
            Unused
        train_x : torch.Tensor
            Unused
        train_obj : torch.Tensor
            Unused

        Returns
        -------
        Tuple (model, likelihood

        """
        mll = ExactMarginalLogLikelihood(likelihood, model)
        fit_gpytorch_mll(mll)
        return model, model.likelihood

    def sample_predictions(self, model, likelihood, model_input, n_samples):
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
            posterior = model.posterior(model_input['x'])
            samples = posterior.sample(sample_size)  # [n_samples, len(test_x), y_dim]
        return samples
