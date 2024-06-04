from typing import List, Optional

from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from rdkit.Chem import MolFromSmiles
import numpy as np
import pandas as pd
# from gauche.dataloader import DataLoaderMP
from gauche.dataloader import MolPropLoader
from gauche.representations.fingerprints import (
    # ecfp_fingerprints,
    fragments,
    mqn_features,
    )
from gauche.representations.strings import bag_of_characters
from gauche.representations.graphs import molecular_graphs
from botied.objectives.base_objective import BaseObjective
from botied.objectives.dummy_problem import DummyGetProblem


class Caco2Plus(BaseObjective):
    _allows_sampling = False
    valid_labels = [
        'Y', 'CrippenClogP', 'TPSA', 'QED', 'ExactMolWt', 'FractionCSP3']
    valid_feature = 'SMILES'
    valid_representations = [
        "ecfp_fingerprints",
        "fragments",
        "ecfp_fragprints",
        "mqn_features",
        "bag_of_characters",
        "molecular_graphs"
    ]

    def __init__(self, botorch_kwargs={}, kwargs={}):
        # botorch_kwargs is not used
        # Must instantiate problem first
        self.problem = DummyGetProblem(kwargs)
        self.path = kwargs['path']
        self.loader = MolPropLoader()
        super(Caco2Plus, self).__init__(botorch_kwargs, kwargs)
        self._load()

    def __len__(self):
        df = self.get_table()
        # Number of non-null rows
        return (~df[self.targets].isnull().any(axis=1).values).sum()

    def _validate(self):
        assert set(self.targets) in set(self.valid_labels)

    def get_table(self):
        if self.path.endswith('.csv'):
            df = pd.read_csv(self.path, index_col=None)
            # Negate to define everything in terms of maximization
            df.loc[:, self.targets] = df[self.targets].values * (
                (np.array(self.modes) == 'max').astype(int)*2 - 1).reshape(
                    -1, len(self.targets))
            # # Drop unused columns
            # df.drop(columns=self.targets[self.num_objectives:], inplace=True)
            return df
        else:
            raise OSError(f'{self.path} extension not supported')

    def _load(self):
        df = self.get_table()
        # drop nans from the datasets
        is_nan = df[self.targets].isnull().any(axis=1).values  # [n_data,]
        features = (
            df[self.valid_feature].iloc[~is_nan].to_list()
        )
        features = self._featurize(features, **self.featurizing_kwargs)
        labels = (
            df[self.targets].iloc[~is_nan].values
        )
        # __getitem__ is defined through problem object
        self.problem.set_data(features, labels)

    def _featurize(
        self,
        features,
        representation,
        bond_radius=3,
        nBits=2048,
        graphein_config=None):
        if representation == "ecfp_fingerprints":
            features = ecfp_fingerprints(
                features, bond_radius=bond_radius, nBits=nBits)
        elif representation == "fragments":
            features = fragments(features)
        elif representation == "ecfp_fragprints":
            features = np.concatenate(
                (
                    ecfp_fingerprints(
                        features, bond_radius=bond_radius, nBits=nBits),
                    fragments(features),
                ),
                axis=1,
            )
        elif representation == "mqn_features":
            features = mqn_features(features)
        elif representation == "bag_of_selfies":
            features = bag_of_characters(features, selfies=True)
        elif representation == "bag_of_smiles":
            features = bag_of_characters(features)
        elif representation == "molecular_graphs":
            features = molecular_graphs(features, graphein_config)
        else:
            raise Exception(
                f"The specified representation choice {representation} is not valid."
                f"Choose from {self.valid_representations}."
            )
        return features


def ecfp_fingerprints(
    smiles: List[str],
    bond_radius: Optional[int] = 3,
    nBits: Optional[int] = 2048,
) -> np.ndarray:
    """
    Builds molecular representation as a binary ECFP fingerprints.

    :param smiles: list of molecular smiles
    :type smiles: list
    :param bond_radius: int giving the bond radius for Morgan fingerprints. Default is 3
    :type bond_radius: int
    :param nBits: int giving the bit vector length for Morgan fingerprints. Default is 2048
    :type nBits: int
    :return: array of shape [len(smiles), nBits] with ecfp featurised molecules

    """

    rdkit_mols = [MolFromSmiles(s) for s in smiles]
    fpgen = GetMorganGenerator(radius=bond_radius, fpSize=nBits)
    fps = [fpgen.GetFingerprint(mol) for mol in rdkit_mols]
    return np.array(fps)


if __name__ == "__main__":
    import os
    import botied
    caco2_plus_path = os.path.join(
        botied.__path__[0], '..', 'data', 'Caco2_Wang_w_props.csv')

    obj = Caco2Plus(
        kwargs={'path': caco2_plus_path,
                'targets':['Y', 'CrippenClogP', 'TPSA'],
                'modes': ['max', 'min', 'max'],
                'ref_point': [-8.0, -9.0, 3.0],
                'featurizing_kwargs': {'representation': 'ecfp_fragprints',},
                'split_frac': {'train': 0.3, 'test': 0.05},
                'negate': False})
    obj.set_split(20)
    init_data = obj.get_initial(n=100)  # n is ignored
    # print((f.max(0)[0] - f.min(0)[0])*0.05)
    print(obj.problem.ref_point)
    print(obj.get_pool(n=20, round_idx=1).keys())
    print(obj.get_pool(n=20, round_idx=1)['x'])  # [29, 2133]
    breakpoint()
