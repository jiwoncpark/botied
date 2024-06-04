from botied.objectives.base_objective import BaseObjective
from botorch.test_functions.multi_objective import BraninCurrin as BraninCurrinBotorch


class BraninCurrin(BaseObjective):
    _allows_sampling = True

    def __init__(self, botorch_kwargs, kwargs):
        # Must instantiate problem first
        self.problem = BraninCurrinBotorch(**botorch_kwargs)
        super(BraninCurrin, self).__init__(botorch_kwargs, kwargs)


if __name__ == "__main__":
    from botorch.utils.sampling import draw_sobol_samples

    obj = BraninCurrin(
        {'negate': True}, {'noise_std': [3.5650, 0.1536], 'ref_point': None})
    x = draw_sobol_samples(bounds=obj.bounds, n=100000, q=1).squeeze(1)
    f = obj.problem(x)
    print(obj.problem.ref_point, obj.problem._ref_point)
    # (f.max(0)[0] - f.min(0)[0])*0.01
    breakpoint()
