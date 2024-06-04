import torch
from botied.objectives.base_objective import BaseObjective
from botorch.test_functions.multi_objective import (
    DTLZ2)  # , DTLZ1, DTLZ3, DTLZ4, DTLZ5, DTLZ7


class DTLZ(BaseObjective):

    _allows_sampling = True
    botorch_keys = ['dim', 'num_objectives', 'noise_std', 'negate']

    def __init__(self, botorch_kwargs, kwargs):
        # Must instantiate problem first
        self.problem = DTLZ2(**botorch_kwargs)
        super(DTLZ, self).__init__(botorch_kwargs, kwargs)
        if 'ref_point' in kwargs:
            sign = -2*float(self.problem.negate) + 1
            self.problem.ref_point = torch.tensor(
                kwargs['ref_point'])*sign

    def set_tkwargs(self, device, dtype):
        self.device = device
        self.dtype = dtype
        self.problem = self.problem.to(
            device=self.device, dtype=self.dtype
        )
        self.problem.noise_std = self.problem.noise_std.to(
            device=self.device, dtype=self.dtype
        )
        self.problem.ref_point = self.problem.ref_point.to(
            device=self.device, dtype=self.dtype
        )


if __name__ == "__main__":
    # import subprocess
    # bashCommand = "strings /homefs/home/parj2/miniforge3/lib/libstdc++.so.6.0.31 | grep GLIBCXX_3.4.29"
    # process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
    # output, error = process.communicate()

    # from scipy.spatial import Delaunay, HalfspaceIntersection
    from botorch.utils.sampling import draw_sobol_samples
    # obj = DTLZ(
    #     botorch_kwargs={'dim': 6, 'num_objectives': 4, 'negate': False},
    #     kwargs={'ref_point': [-0.1, -0.1, -0.1, -0.1]})
    # # x = draw_sobol_samples(bounds=obj.problem.bounds, n=100000, q=1).squeeze(1)
    # # f = obj.problem(x)
    # # print((f.max(0)[0] - f.min(0)[0])*0.05)
    # print(obj.problem.ref_point)
    # # print(obj.gen_pareto_front(10))
    # breakpoint()

    obj = DTLZ(
        botorch_kwargs={'dim': 7, 'num_objectives': 6, 'negate': False},
        kwargs={'ref_point': [-0.1]*6})
    x = draw_sobol_samples(bounds=obj.problem.bounds, n=100000, q=1).squeeze(1)
    f = obj.problem(x)
    print("min", f.min(0)[0])
    print("max", f.max(0)[0])
    print((f.max(0)[0] - f.min(0)[0])*0.01)
    print(obj.problem.ref_point)
    print(obj.problem.gen_pareto_front(10))
    breakpoint()
