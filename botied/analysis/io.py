import pandas as pd


def load_metadata(metadata_path, exclude_seeds=[]):
    meta = pd.read_csv(metadata_path, index_col=None)
    meta = meta[meta.State.isin(['finished'])]
    meta = meta[~meta.seed.isin(exclude_seeds)]
    meta = meta.sort_values(by='seed').reset_index(drop=True)
    return meta
