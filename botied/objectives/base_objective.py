from abc import ABC  # , abstractmethod
import math
import copy
from typing import Optional, List, Dict
import numpy as np
import torch
from botorch.utils.sampling import draw_sobol_samples
from botorch.utils.transforms import normalize, standardize, unnormalize
import logging
logger = logging.getLogger(__name__)


class BaseObjective(ABC):
    """Base objective class

    """
    _allows_sampling: bool = NotImplemented  # whether we can sample designs

    def __init__(
            self, botorch_kwargs: dict = {}, kwargs: dict = {}):
        """
        Parameters
        ----------
        options
        """
        for key in kwargs:
            setattr(self, key, kwargs[key])
        # Override default object variables
        if 'noise_std' in kwargs:
            self._set_noise_std(kwargs.get('noise_std'))
        else:
            self.problem.noise_std = torch.zeros(self.problem.num_objectives)
        # self._set_bounds(kwargs.get('bounds'))
        # self._set_ref_point(kwargs.get('ref_point'))
        # Negate default ref point
        # sign: -1 if negate=True, 1 if negate=False
        # if 'negate' in botorch_kwargs:
        #     sign = -2.0*int(botorch_kwargs.get('negate')) + 1
        # else:
        #     sign = 1.0
        # Choose initial number of points
        if self._allows_sampling:
            self.initial_n = 2*(self.problem.dim+1)
        else:
            self.initial_n = None
        clean_prob = copy.deepcopy(self.problem)
        clean_prob.noise_std = None
        self.clean_prob = clean_prob

    def set_tkwargs(self, device, dtype):
        self.device = device
        self.dtype = dtype
        self.problem = self.problem.to(
            device=self.device, dtype=self.dtype
        )
        self.problem.noise_std = self.problem.noise_std.to(
            device=self.device, dtype=self.dtype
        )

    def _set_noise_std(self, noise_std: Optional[list]):
        if noise_std is not None:
            logger.info(f"Setting noise_std to {noise_std}")
            self.problem.noise_std = torch.tensor(noise_std)
        else:
            self.problem.noise_std = torch.zeros(self.problem.num_objectives)

    def _set_bounds(self, bounds: Optional[list]):
        """
        Unused for our experiments. Use bounds in botorch test_functions.
        """
        if bounds is not None:
            logger.info(f"Setting bounds to {bounds}")
            self.problem.bounds = torch.tensor(bounds)

    def concat_input(self, data_list: List[Dict]):
        """Default way to concat just the input

        """
        # Additionally compute 'scaled_x'
        out = {'x': torch.cat([d['x'] for d in data_list], dim=0)}
        if self._allows_sampling:
            out['scaled_x'] = normalize(out['x'], self.problem.bounds)
        else:
            out['scaled_x'] = out['x']
        return out

    # def concat_all(self, data_list: List[Dict], collate=True):
    #     out = {}
    #     for k in ['x', 'y', 'clean_y']:
    #         out[k] = torch.cat([d[k] for d in data_list], dim=0)
    #     return out

    def concat(
            self, data_list: List[Dict],
            collate: bool = True, to_numpy: bool = False):
        """Default way to concat labels over rounds

        """
        keys = ['y', 'clean_y']
        out_dict = self.concat_input(data_list)
        for k in keys:
            if collate:
                concat = torch.cat([d[k] for d in data_list], dim=0)
                if to_numpy:
                    concat = concat.detach().cpu().numpy()
            else:
                if to_numpy:
                    concat = [d[k].detach().cpu().numpy() for d in data_list]
                else:
                    concat = [d[k] for d in data_list]
            out_dict.update({k: concat})
            # Additionally compute 'scaled_y'
            if k == 'y' and not to_numpy and collate:
                out_dict.update({'scaled_y': standardize(concat)})
        return out_dict

    def slice(self, data, indices, collate: bool = True):
        """Default way to slice a portion of the dataset based on index

        """
        out_data = {
            'x': data['x'][indices],
            'y': data['y'][indices],
            'clean_y': data['clean_y'][indices]}
        return out_data

    def evaluate(self, x):
        """
        x: normalized x (normalized to [0, 1]) from optimizer

        """
        x = unnormalize(x, self.problem.bounds)
        clean_y = self.clean_prob(x)
        noise = torch.randn_like(clean_y) * self.problem.noise_std  # [N, 2]
        y = clean_y + noise
        return {
            # 'x': normalize(x, self.problem.bounds),
            'x': x,
            # 'y': standardize(y),
            'y': y,
            'clean_y': clean_y}

    def set_split(self, n_rounds):
        """Splits data into training, validation, and pool sets

        """
        if self._allows_sampling:
            return
        # Computes split distributions
        n_data = len(self)
        n_train = math.ceil(n_data*self.split_frac['train'])
        n_test = math.ceil(n_data*self.split_frac['test'])
        n_pool = n_data - n_train - n_test
        n_pool_per_round = math.floor(n_pool/n_rounds)
        idx = np.arange(n_data)
        np.random.shuffle(idx)  # inplace
        sizes = [n_train, n_test] + [n_pool_per_round]*n_rounds
        names = ['train', 'test'] + [f'round_{r}_pool' for r in range(1, n_rounds+1)]
        cut_idx = np.cumsum(sizes)
        split_idx_list = np.split(idx, cut_idx)[:-1]
        split_idx_dict = dict(zip(names, split_idx_list))
        self.split = split_idx_dict

    def get_initial(self, n: int = None, collate: bool = True):
        if self._allows_sampling:
            if n is None:
                n = self.initial_n
            return self.simulate(n)
        else:
            if n is not None:
                logger.warning(
                    "`n` argument ignored, instead using"
                    " `objective.split_frac.train` for initial points.")
            return self.slice(
                self.problem, self.split['train'], collate=collate)
            # return Batch.from_data_list(
            #     [self.problem[i] for i in self.split['train']],
            #     False, False)
            # return collate(
            #     self.problem[0].__class__,
            #     data_list=[self.problem[i] for i in self.split['train']],
            #     increment=False,
            #     add_batch=False)

    def get_pool(self, n: int = None, round_idx: int = None):
        if self._allows_sampling:
            return self.simulate(n)
        else:
            if not hasattr(self, 'split'):
                raise ValueError("`set_split` must be called first.")
            # return collate(
            #     self.problem[0].__class__,
            #     data_list=[self.problem[i] for i in self.split[f'round_{round_idx}_pool']],
            #     increment=False,
            #     add_batch=False)
            # return Batch.from_data_list(
            #     [self.problem[i] for i in self.split[f'round_{round_idx}_pool']],
            #     False, False)
            logger.warning("`n` ignored, fetching pre-sliced pool.")
            return self.slice(self.problem, self.split[f'round_{round_idx}_pool'])

    def simulate(self, n: int = None):
        """
        Only called when _allows_sampling is True
        """
        if n is None:
            n = 2*(self.problem.dim+1)
        x = draw_sobol_samples(
            bounds=self.problem.bounds, n=n, q=1).squeeze(1).to(
                dtype=self.dtype, device=self.device)  # [N, 2]
        clean_y = self.clean_prob(x)
        noise = torch.randn_like(clean_y) * self.problem.noise_std  # [N, 2]
        y = clean_y + noise
        out_dict = dict(
            # x=normalize(x, self.problem.bounds),
            x=x,
            # y=standardize(y),
            y=y,
            clean_y=clean_y
        )
        return out_dict
