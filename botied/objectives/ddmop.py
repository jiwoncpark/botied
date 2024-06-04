import numpy as np
import pandas as pd
from botied.objectives.base_objective import BaseObjective
from botied.objectives.dummy_problem import DummyGetProblem


class DDMOP(BaseObjective):
    _allows_sampling = False

    def __init__(self, botorch_kwargs, kwargs):
        # botorch_kwargs is not used
        # Must instantiate problem first
        self.problem = DummyGetProblem(kwargs)

        self.ddmop_id = str(7)  # kwargs['ddmop_id'])
        self.s3_path = 's3://prescient-data-dev/sandbox/tagasovn/ddmop_data/'
        self.path_features = self.s3_path + 'ddmop' + self.ddmop_id + '_in.csv'
        self.path_labels = self.s3_path + 'ddmop' + self.ddmop_id + '_out.csv'
        self.path_df = self.s3_path + 'ddmop' + self.ddmop_id + '_df.csv'
        super(DDMOP, self).__init__(botorch_kwargs, kwargs)
        self._load()

    def __len__(self):
        df = np.asarray(pd.read_csv(self.path_features, header=None))
        return df.shape[0]

    def _validate(self):
        assert set(self.targets) in set(self.valid_labels)

    def _load(self):
        table = pd.read_csv(self.path_df)
        features = np.asarray(pd.read_csv(self.path_features, header=None))
        labels = np.asarray(pd.read_csv(self.path_labels, header=None))
        self.targets = table.columns[table.columns.str.startswith('t')]

        self.problem.set_data(features, labels)


if __name__ == "__main__":

    botorch_kwargs = {"root": None}
    kwargs = {
        "initial_n": 10,
        "ddmop_id": 1,
        "modes": ["max"]*11,
        "bounds": [[21.227, -0.77466, -4.197, -0.20425, -0.37559, -69.639, 3.6346, 3.9993, 13.328, 0.0, 0.0],
                   [40.08, 2.1368, 4.4888, 0.59861, 0.92812, 183.81, 4.3505, 13.495, 15.874, 0.0, 0.0]],
        "ref_point": [
            21.226,
            -0.77566,
            -4.198,
            -0.20525,
            -0.37659,
            -69.64,
            3.6336,
            3.9983,
            13.327],
        "split_frac": {"train": 0.05, "test": 0.2},
        "num_objectives": 9,
        "negate": True,
    }

    ddmop = DDMOP(botorch_kwargs, kwargs)
    breakpoint()
