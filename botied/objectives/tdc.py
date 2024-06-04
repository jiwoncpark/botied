from typing import List
import os
import numpy as np
import pandas as pd
import torch
import torch_geometric
# from torch_geometric.data.collate import collate
from torch_geometric.data.batch import Batch
from torch_geometric.data import Dataset
from rdkit import Chem
from botied.objectives.base_objective import BaseObjective
import logging
logger = logging.getLogger(__name__)


class TDC(BaseObjective):

    _allows_sampling = False

    def __init__(self, botorch_kwargs, kwargs):
        # Must instantiate problem first
        self.problem = TDCProblem(botorch_kwargs)
        super(TDC, self).__init__(botorch_kwargs, kwargs)

    def __len__(self):
        return self.problem.n_data

    def concat_input(self, data_list: List, collate: bool):
        concat_data = torch.utils.data.ConcatDataset(data_list)
        # FIXME: only the inputs
        if collate:
            return Batch.from_data_list(concat_data, False, False)
        return concat_data

    def concat(self, data_list: List, collate: bool = True):
        concat_data = torch.utils.data.ConcatDataset(data_list)
        if collate:
            return Batch.from_data_list(concat_data, False, False)
        return concat_data

    def slice(self, data, indices, collate: bool = True):
        if not isinstance(indices, list):
            indices = indices.tolist()  # for np.ndarray or torch.tensor
        subset = torch.utils.data.Subset(data, indices)
        if collate:
            return Batch.from_data_list(subset, False, False)
        return subset


class TDCProblem(Dataset):
    dim = 10  # arbitrary # FIXME
    # TODO: put required kwargs

    def __init__(self, botorch_kwargs):
        for key in botorch_kwargs:
            setattr(self, key, botorch_kwargs[key])
        if 'root' not in botorch_kwargs:
            self.root = 'caco2'
        if '_noise_std' not in botorch_kwargs:
            self._noise_std = torch.zeros(self.num_objectives)
        # Define metadata
        table = self.get_table()
        self.objectives = self.targets[:self.num_objectives]
        self._ref_point = table[self.objectives].min().values
        # Find number of non-None graphs  FIXME: processing runs twice
        n_data = 0
        for _, row in table.iterrows():
            graph = row_to_graph(
                row, self.atom_types, self.bond_types, self.objectives)
            if graph is not None:
                n_data += 1
        self.n_data = n_data
        super(TDCProblem, self).__init__(
            root=self.root)
        self.device = torch.device('cpu')
        self.dtype = torch.float64

    def to(self, device, dtype):
        self.device = device
        self.dtype = dtype
        return self

    @property
    def raw_file_names(self):
        return [self.path]

    @property
    def processed_file_names(self):
        return [f'data_{i}.pt' for i in range(self.n_data)]

    def len(self):
        return self.n_data  # 901

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'data_{idx}.pt'))
        data.y = data.clean_y + (
            torch.randn(self.num_objectives)*self._noise_std).unsqueeze(0)
        # TODO: cast to self.dtype
        return data.to(self.device)

    def get_table(self):
        if self.path.endswith('.csv'):
            df = pd.read_csv(self.path, index_col=None)
            # Negate to define everything in terms of maximization
            df.loc[:, self.targets] = df[self.targets].values * (
                (np.array(self.modes) == 'max').astype(int)*2 - 1).reshape(
                    -1, len(self.targets))
            # Drop unused columns
            df.drop(columns=self.targets[self.num_objectives:], inplace=True)
            return df
        else:
            raise OSError(f'{self.path} extension not supported')

    def process(self):
        """Generates 3d conformer graphs

        Returns:
            graphs: Output DataFrame with graphs
        """
        table = self.get_table()
        idx = 0
        for _, row in table.iterrows():
            graph = row_to_graph(
                row, self.atom_types, self.bond_types, self.objectives)
            if graph is not None:
                torch.save(graph, os.path.join(
                    self.processed_dir, f'data_{idx}.pt'))
                idx += 1


def row_to_graph(
        row: pd.Series, atom_types, bond_types, objectives) -> torch_geometric.data.Data:
    """Builds graph from molecule data

    Args:
        row: Series with molecule data
        config: Experiment config dict
    Returns:
        graph: pyg graph with molecule data
    """
    # Catch-all for any failed steps

    smiles = row['SMILES']
    mol = Chem.MolFromSmiles(row['SMILES'])

    # Catches failed mol parsing
    if mol is None or any([n == 0 for n in [mol.GetNumAtoms(),
                                            mol.GetNumBonds()]]):
        return

    # Sets up empty atom arrays
    x = np.zeros(
        [mol.GetNumAtoms(), len(atom_types)])  # Using sparse tensor type

    # Fills in atom data
    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()
        symbol = atom.GetSymbol()
        # Filters molecules with atoms not in provided types
        if symbol not in atom_types:
            return
        x[atom_idx][atom_types.index(symbol)] = 1

    # Fills in bond data
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        b, e = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index.extend([[b, e], [e, b]])
        bond_type = bond.GetBondType().name
        # Filters molecules with bonds not in provided types
        if bond_type not in bond_types:
            return
        edge_attr.extend([bond_types.index(bond_type)]*2)

    # Gets label data
    y = row[objectives].to_numpy(dtype=np.float64)

    # Converts data to tensors
    x = torch.tensor(x, dtype=torch.float)
    edge_index = torch.tensor(
        np.array(edge_index), dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(np.array(edge_attr), dtype=torch.float)
    y = torch.tensor(y, dtype=torch.float)

    # Constructs graph
    graph = torch_geometric.data.Data(
        x=x, edge_index=edge_index, edge_attr=edge_attr,
        clean_y=y.unsqueeze(0),  # [1, self.num_objectives]
        SMILES=smiles)

    # Adds pre-assigned split
    split = None if 'split' not in row else row['split']
    if split is not None:
        graph['split'] = split

    # # Catches all remaining mol/graph failures
    # except:
    #     graph = None

    return graph
