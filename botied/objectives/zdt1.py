from botied.objectives.base_objective import BaseObjective
from botorch.test_functions.multi_objective import ZDT1 as ZDT1Botorch


class ZDT1(BaseObjective):
    _allows_sampling = True

    def __init__(self, botorch_kwargs, kwargs):
        # Must instantiate problem first
        self.problem = ZDT1Botorch(**botorch_kwargs)
        super(ZDT1, self).__init__(botorch_kwargs, kwargs)


if __name__ == "__main__":
    from botorch.utils.sampling import draw_sobol_samples

    obj = ZDT1(
        botorch_kwargs={'negate': True, "num_objectives": 2, "dim": 4},
        kwargs={'noise_std': None, 'ref_point': None})
    x = draw_sobol_samples(bounds=obj.problem.bounds, n=1000, q=1).squeeze(1)
    f = obj.problem(x)
    print(obj.problem.bounds)
    print("max", f.max(0)[0])  # [-0.0004, -0.3024]
    print("min", f.min(0)[0])  # [-0.9998, -8.5524]
    print(r"1%", (f.max(0)[0] - f.min(0)[0])*0.01)  # [0.0100, 0.0763]
    print(r"5%", (f.max(0)[0] - f.min(0)[0])*0.05)  # [0.0500, 0.4125]
    print(obj.problem.ref_point)  # [-11., -11.] by default
    breakpoint()
