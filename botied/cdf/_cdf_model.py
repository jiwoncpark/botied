import numpy as np
import pyvinecopulib as pv
from scipy.stats import multivariate_normal
from statsmodels.nonparametric.kernel_density import KDEMultivariate
import logging
logger = logging.getLogger(__name__)


class CDFModel:
    def __init__(
            self,
            type: str = 'copula',
            init_kwargs: dict = {
                'family_set_names': ['tll'], 'nonparametric_mult': 0.001},
            eval_kwargs: dict = {
                'N': 1000, 'num_threads': 4},
            ties_method="random"):
        if type not in ['copula', 'copula_inverse', 'mvn', 'empirical', 'kde']:
            raise ValueError(
                "Must be one of ['copula', 'mvn', 'empirical', 'kde']")
        self.type = type
        self.ties_method = ties_method
        # type == 'copula
        self.family_set_names = init_kwargs.get('family_set_names')
        if self.family_set_names == "None":
            self.controls = pv.FitControlsVinecop()
        else:
            family_set = [
                getattr(pv.BicopFamily, fam) for fam in self.family_set_names]
            self.nonparametric_mult = init_kwargs.get('nonparametric_mult')
            if 'tll' not in self.family_set_names:
                if self.nonparametric_mult is not None:
                    logger.warning("`nonparametric_mult` provided but ignored.")
                self.controls = pv.FitControlsVinecop(family_set=family_set)
            else:
                self.controls = pv.FitControlsVinecop(
                    family_set=family_set,
                    nonparametric_mult=self.nonparametric_mult
                )
        self.eval_kwargs = eval_kwargs

    def __call__(self, y, eval_y=None):
        """
        y : np.ndarray
            Data of shape [num_points, y_dim]

        """
        # Negate objectives
        y = -y
        if eval_y is not None:
            eval_y = -eval_y
        return 1.0 - getattr(self, f'_eval_{self.type}')(y, eval_y)

    def _eval_copula(self, y, eval_y):
        u = pv.to_pseudo_obs(y, ties_method=self.ties_method)
        copula = pv.Vinecop(u, controls=self.controls)
        eval_u = pv.to_pseudo_obs(
            eval_y, ties_method=self.ties_method) if eval_y is not None else u
        cdf_scores = copula.cdf(
            eval_u, **self.eval_kwargs)  # [len(y)]
        return cdf_scores

    def _eval_copula_inverse(self, y, eval_y):
        """
        Fit CDF on y to obtain F_Y where Y are max obj
        and score by F_Y
        (returns 1 - F_Y but reversed by __call__)
        """
        u = pv.to_pseudo_obs(-y, ties_method=self.ties_method)
        copula = pv.Vinecop(u, controls=self.controls)
        eval_u = pv.to_pseudo_obs(
            -eval_y, ties_method=self.ties_method) if eval_y is not None else u
        cdf_scores = copula.cdf(
            eval_u, **self.eval_kwargs)  # [len(y)]
        return 1.0 - cdf_scores

    def _eval_mvn(self, y, eval_y):
        empirical_mean = np.mean(y, axis=0)
        empirical_cov = np.cov(y, rowvar=0)
        # Add numerical buffer
        empirical_cov[np.diag_indices_from(empirical_cov)] = empirical_cov[
            np.diag_indices_from(empirical_cov)] + 1.e-5
        eval_y = y if eval_y is None else eval_y
        cdf_scores = multivariate_normal.cdf(
            eval_y, mean=empirical_mean, cov=empirical_cov)
        return cdf_scores

    def _eval_empirical(self, y, eval_y):
        y = np.atleast_2d(y)
        eval_y = y if eval_y is None else np.atleast_2d(eval_y)
        ecdf = np.empty(eval_y.shape[0])  # [N,]
        for i, eval_row in enumerate(eval_y):
            # row ~ [M,], y ~ [N, M]
            ecdf[i] = (y <= eval_row).all(1).mean()
        return ecdf

    def _eval_kde(self, y, eval_y):
        data_type = np.repeat('c', y.shape[1])
        data_kde = KDEMultivariate(y, var_type=data_type, bw="cv_ml")
        eval_y = y if eval_y is None else eval_y
        cdf_scores = data_kde.cdf(eval_y)
        return cdf_scores


if __name__ == "__main__":
    # copula
    cop_cdf = CDFModel()
    y = np.random.randn(100, 3)
    cop_cdf_scores = cop_cdf(y)
    # mvn
    mvn_cdf = CDFModel(type='mvn')
    mvn_cdf_scores = mvn_cdf(y)
    # empirical
    emp_cdf = CDFModel(type='empirical')
    emp_cdf_scores = emp_cdf(y)
    # kde
    kde_cdf = CDFModel(type='kde')
    kde_cdf_scores = kde_cdf(y)
    print(
        cop_cdf_scores.shape,
        mvn_cdf_scores.shape,
        emp_cdf_scores.shape,
        kde_cdf_scores.shape)
