import torch
from omegaconf import OmegaConf


class BaseDummyProblem:
    def __init__(self, kwargs):
        if OmegaConf.is_dict(kwargs):
            kwargs = OmegaConf.to_container(kwargs, resolve=True)

        if 'num_objectives' in kwargs:
            self.num_objectives = kwargs['num_objectives']
            if kwargs.get('targets') is not None:
                if len(kwargs.get('targets')) != self.num_objectives:
                    raise ValueError(
                        "Specified list of targets does not agree with"
                        f" specified num_objectives={self.num_objectives}")
        else:
            self.num_objectives = len(kwargs['targets'])

        if 'noise_std' in kwargs:
            self.noise_std = torch.tensor(kwargs['noise_std'])
        else:
            self.noise_std = torch.zeros(self.num_objectives)

        if 'ref_point' not in kwargs:
            raise ValueError("`ref_point` must be provided.")
        self.ref_point = torch.tensor(kwargs['ref_point'])

        if 'bounds' in kwargs:
            self.bounds = torch.tensor(kwargs['bounds'])
        else:
            self.bounds = None

    def to(self, device: torch.device, dtype: torch.dtype):
        self.ref_point = self.ref_point.to(device=device, dtype=dtype)
        self.noise_std = self.noise_std.to(device=device, dtype=dtype)
        if self.bounds is not None:
            self.bounds = self.bounds.to(device=device, dtype=dtype)

class DummySampleProblem(BaseDummyProblem):
    def __init__(self, kwargs):
        super(DummySampleProblem, self).__init__(kwargs)

    def __call__(self):
        pass


class DummyGetProblem(BaseDummyProblem):
    def __init__(self, kwargs):
        super(DummyGetProblem, self).__init__(kwargs)

    def set_data(self, features, labels):
        self.features = features
        self.labels = labels

    def __getitem__(self, key):
        if key == 'x':
            return self.features
        elif key in ['y', 'clean_y']:
            return self.labels

    def to(self, device: torch.device, dtype: torch.dtype):
        self.ref_point = self.ref_point.to(device=device, dtype=dtype)
        self.noise_std = self.noise_std.to(device=device, dtype=dtype)
        if self.bounds is not None:
            self.bounds = self.bounds.to(device=device, dtype=dtype)
        setattr(
            self, 'features', torch.tensor(self.features).to(dtype).to(device))
        setattr(
            self, 'labels', torch.tensor(self.labels).to(dtype).to(device))
        return self
