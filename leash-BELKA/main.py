# %% [markdown]
"""
# Leash BELKA
"""

# %%
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data

TRAIN_CSV = "./data/train.csv"
TEST_CSV = "./data/test.csv"
TRAIN_PARQUET = "./data/train.parquet"
TEST_PARQUET = "./data/test.parquet"

# %%
hyperparameters = {}

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

# %%
SEED = 521
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# %%
PROTIEN_NAME_TO_TARGET = {
    "sEH": 0,
    "BRD4": 1,
    "ALB": 2,
    "HSA": 2,
}


class LeashBelkaDataset(data.Dataset):
    def __init__(self, filename: str) -> None:
        super(LeashBelkaDataset).__init__()

        self.filename = filename
        self.columns = self.get_columns(filename)
        self.num_samples = self.get_num_samples(filename)

    def __getitem__(self, idx: int) -> Dict:
        dataframe = self.read_partial_dataframe_from_csv(idx)

        try:
            molecule_smiles = dataframe["molecule_smiles"].tolist()[0]
            protien_name = dataframe["protein_name"].tolist()[0]
            binds = int(dataframe["binds"].tolist()[0])
        except KeyError as e:
            print(
                "Error occurred while decoding dataframe. "
                f"Could not find column with key {e}."
            )
            exit(-1)

        return {
            "smiles": molecule_smiles,
            "target": torch.tensor(PROTIEN_NAME_TO_TARGET[protien_name]),
            "binds": torch.tensor(binds),
        }

    def __len__(self) -> int:
        return self.num_samples

    def read_partial_dataframe_from_csv(self, idx: int, chunksize: int = 1):
        csv_reader = pd.read_csv(
            self.filename,
            chunksize=chunksize,
            skiprows=idx + 1,
            low_memory=True,
            header=None,
            names=self.columns,
        )
        dataframe = next((chunk for chunk in csv_reader))
        return dataframe

    @staticmethod
    def get_num_samples(filename: str):
        return sum(1 for _ in open(filename)) - 1

    @staticmethod
    def get_columns(filename: str):
        csv_reader = pd.read_csv(
            filename,
            chunksize=1,
            low_memory=True,
        )
        dataframe = next((chunk for chunk in csv_reader))
        return dataframe.columns


# %%
class LeashBelkaDataLoader(data.DataLoader):
    def __init__(self, dataset: LeashBelkaDataset, *args, **kwargs) -> None:
        super().__init__(
            dataset=dataset,
            collate_fn=self.collate_fn,
            *args,
            **kwargs,
        )

    @staticmethod
    def collate_fn(batch):
        smiles = [item.get("smiles") for item in batch]
        target = torch.concat(
            [item.get("target").unsqueeze(dim=0) for item in batch],
        )
        binds = torch.concat(
            [item.get("binds").unsqueeze(dim=0) for item in batch],
        )

        return {"smiles": smiles, "target": target, "binds": binds}


# %%
dataset = LeashBelkaDataset(TRAIN_CSV)
dataloader = LeashBelkaDataLoader(dataset, batch_size=32)
for idx, batch in enumerate(dataloader):
    print(batch)
