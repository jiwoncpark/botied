import numpy as np
import torch
from scipy.stats import beta
from botied.objectives.base_objective import BaseObjective
from botied.objectives.dummy_problem import DummyGetProblem
import pyvinecopulib as pv
from botorch.test_functions.multi_objective import BraninCurrin as BraninCurrinBotorch


class CopulaBraninCurrin(BaseObjective):
    _allows_sampling = False
    r"""
    Generative procedure:
    1. Sample u ~ clayton copula
    2. Transform margins with y = beta ppf(u)
    2. Map x = BC(y)

    So Y = BC^{-1}(X) ~ Gaussian copula

    """

    def __init__(self, botorch_kwargs, kwargs):
        # Must instantiate problem first
        self.problem = DummyGetProblem(kwargs)
        self.inverse_problem = BraninCurrinBotorch(
            **botorch_kwargs.inverse_kwargs)
        super(CopulaBraninCurrin, self).__init__(botorch_kwargs, kwargs)
        self._load()

    def __len__(self):
        return self.num_samples

    def _load(self):
        features, labels = self.generate(self.num_samples)
        self.problem.set_data(features, labels)

    def generate(self, num_samples):
        # FIXME: make copula family configurable
        clay_cop = pv.Bicop(
            family=pv.BicopFamily.clayton,
            rotation=self.rotation,  # for inverse, but make configurable
            parameters=[1])
        u = clay_cop.simulate(num_samples, seeds=[1])
        y = np.asarray(
            [beta.ppf(u[:, i], a=2.0, b=2.0) for i in range(0, 2)]
        ).squeeze().transpose()  # y marginals ~ beta
        x = self.inverse_problem(torch.tensor(y))
        return x, y
