"""
Source:
https://github.com/pyg-team/pytorch_geometric/blob/master/torch_geometric/loader/dataloader.py

Minor change to the Collater class such that the dataloader returns batches in
the form (batch, y) instead of (batch).
"""

from collections.abc import Mapping, Sequence
from typing import Any

import torch.utils.data
from torch.utils.data.dataloader import default_collate
from torch_geometric.data import Batch, Dataset, Data
from torch_geometric.data.data import BaseData
from torch_geometric.data.datapipes import DatasetAdapter
from torch_geometric.typing import TensorFrame, torch_frame
from torch_geometric.transforms import BaseTransform


import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform

class GetTarget(BaseTransform):
    def __init__(self, target: int | None = None) -> None:
        self.target = [target]

    def forward(self, data: Data) -> Data:
        if self.target is not None:
            data.y = data.y[:, self.target]
        return data
    
    
class ConvertTargetType(BaseTransform):
    """
    A transform to select a target column (if specified)
    and convert the target tensor 'y' to a specific dtype.
    """

    def __init__(self, target: int | None = None, dtype: torch.dtype = torch.float):
        """
        Args:
            target (int | None, optional): The index of the target column
                to select. If None, the entire 'data.y' tensor is converted.
                Defaults to None.
            dtype (torch.dtype, optional): The desired data type
                (e.g., torch.long, torch.float). Defaults to torch.float.
        """
        super().__init__()
        self.target = target
        self.dtype = dtype

    def forward(self, data: Data) -> Data:
        # Check if a specific target column is requested
        if self.target is not None:
            # Select the column and convert its type
            data.y = data.y[:, [self.target]].to(self.dtype)
        else:
            # Convert the entire 'y' tensor
            data.y = data.y.to(self.dtype)
        return data


class ConvertFeaturesToFloat(BaseTransform):
    """Converts node features 'x' to torch.float."""
    def __init__(self):
        super().__init__()

    def forward(self, data: Data) -> Data:
        if data.x.dtype != torch.float:
            data.x = data.x.float()
        return data


class TwoAugmentationsTransform(BaseTransform):
    """Creates two augmented versions of the input data object."""
    def __init__(self, transform: BaseTransform):
        super().__init__()
        self.transform = transform

    def forward(self, data: Data) -> tuple[Data, Data]:
        data1 = self.transform(data)
        data2 = self.transform(data)
        return data1, data2







class Collater:
    def __init__(
        self,
        dataset: Dataset | Sequence[BaseData] | DatasetAdapter,
        follow_batch: list[str] | None = None,
        exclude_keys: list[str] | None = None,
    ):
        self.dataset = dataset
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def __call__(self, batch: list[Any]) -> Any:
        elem = batch[0]
        if isinstance(elem, BaseData):
            batch = Batch.from_data_list(
                batch,
                follow_batch=self.follow_batch,
                exclude_keys=self.exclude_keys,
            )
            # return (batch, batch.y)
            return batch  # Modified to return only batch
        elif isinstance(elem, torch.Tensor):
            return default_collate(batch)
        elif isinstance(elem, TensorFrame):
            return torch_frame.cat(batch, dim=0)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, str):
            return batch
        elif isinstance(elem, Mapping):
            return {key: self([data[key] for data in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, "_fields"):
            return type(elem)(*(self(s) for s in zip(*batch)))
        elif isinstance(elem, Sequence) and not isinstance(elem, str):
            return [self(s) for s in zip(*batch)]

        raise TypeError(f"DataLoader found invalid type: '{type(elem)}'")


class DataLoader(torch.utils.data.DataLoader):
    r"""A data loader which merges data objects from a
    :class:`torch_geometric.data.Dataset` to a mini-batch.
    Data objects can be either of type :class:`~torch_geometric.data.Data` or
    :class:`~torch_geometric.data.HeteroData`.

    Args:
        dataset (Dataset): The dataset from which to load the data.
        batch_size (int, optional): How many samples per batch to load.
            (default: :obj:`1`)
        shuffle (bool, optional): If set to :obj:`True`, the data will be
            reshuffled at every epoch. (default: :obj:`False`)
        follow_batch (List[str], optional): Creates assignment batch
            vectors for each key in the list. (default: :obj:`None`)
        exclude_keys (List[str], optional): Will exclude each key in the
            list. (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch.utils.data.DataLoader`.
    """

    def __init__(
        self,
        dataset: Dataset | Sequence[BaseData] | DatasetAdapter,
        batch_size: int = 1,
        shuffle: bool = False,
        follow_batch: list[str] | None = None,
        exclude_keys: list[str] | None = None,
        **kwargs,
    ):
        # Remove for PyTorch Lightning:
        kwargs.pop("collate_fn", None)

        # Save for PyTorch Lightning < 1.6:
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        super().__init__(
            dataset,
            batch_size,
            shuffle,
            collate_fn=Collater(dataset, follow_batch, exclude_keys),
            **kwargs,
        )
