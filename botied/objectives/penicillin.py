from botied.objectives.base_objective import BaseObjective
from botorch.test_functions.multi_objective import Penicillin as PenicillinBotorch


class Penicillin(BaseObjective):
    _allows_sampling = True

    def __init__(self, botorch_kwargs, kwargs):
        # Must instantiate problem first
        self.problem = PenicillinBotorch(**botorch_kwargs)
        super(Penicillin, self).__init__(botorch_kwargs, kwargs)


if __name__ == "__main__":
    from botorch.utils.sampling import draw_sobol_samples

    obj = Penicillin(
        botorch_kwargs={'negate': True}, kwargs={'noise_std': None, 'ref_point': None})
    x = draw_sobol_samples(bounds=obj.problem.bounds, n=1000, q=1).squeeze(1)
    f = obj.problem(x)
    print((f.max(0)[0] - f.min(0)[0])*0.01)  # [0.1411, 0.7889, 3.9600]
    print(obj.problem.ref_point)  # [  -1.8500,  -86.9300, -514.7000]
    breakpoint()
