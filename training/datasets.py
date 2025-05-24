import os
import numpy as np
import torch
from torch.utils.data import Dataset


class SelectorDataset(Dataset):
    """
    Dataset for selector model:
      - loads .npz files where 'features' is (N, F)
        and 'target' is an integer index (0 <= idx < N).
    """
    def __init__(self, data_dir: str):
        self.files = [os.path.join(data_dir, f)
                      for f in os.listdir(data_dir) if f.endswith('.npz')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        data = np.load(self.files[idx])
        features = data['features'].astype(np.float32)  # (N, F)
        target = int(data['target'].astype(np.int64))   # scalar index
        return torch.from_numpy(features), torch.tensor(target)


class RegressorDataset(Dataset):
    """
    Dataset for regressor model:
      - loads .npz files where 'features' is (N, F)
        and 'target' is (N,) float array of step-sizes.
    """
    def __init__(self, data_dir: str):
        self.files = [os.path.join(data_dir, f)
                      for f in os.listdir(data_dir) if f.endswith('.npz')]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        data = np.load(self.files[idx])
        features = data['features'].astype(np.float32)  # (N, F)
        target = data['target'].astype(np.float32)       # (N,)
        return torch.from_numpy(features), torch.from_numpy(target)


if __name__ == '__main__':
    # Sanity check for datasets
    sd = SelectorDataset('training/data/selector')
    f, t = sd[0]
    print(f"Selector sample features shape: {f.shape}, target idx: {t}")

    rd = RegressorDataset('training/data/regressor')
    f2, t2 = rd[0]
    print(f"Regressor sample features shape: {f2.shape}, target shape: {t2.shape}")
