from sklearn.model_selection import train_test_split
import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.data.dataloader import default_collate
import traceback
import h5py

def multi_import(data_dir,label_name,val_size,batch_size):
    class HeartbeatDataset(Dataset):
        def __init__(self, file_list, label_list):
            self.file_list = file_list
            self.label_list = label_list

        def __len__(self):
            return len(self.file_list)

        def __getitem__(self, idx):
            data = np.load(self.file_list[idx])
            label = self.label_list[idx].astype(np.float32)
            return torch.from_numpy(data).float(), torch.from_numpy(label)  # 将数据转换为 FloatTensor

    class NewHeartbeatDataset(HeartbeatDataset):
        def __init__(self, file_list, label_list):
            super(NewHeartbeatDataset, self).__init__(file_list, label_list)

        def __getitem__(self, idx):
            try:
                return super(NewHeartbeatDataset, self).__getitem__(idx)
            except Exception as e:
                print(f"Error in loading data at index {idx}: {e}")
                print(traceback.format_exc())
                return None, None

    def heartbeat_collate_fn(batch):
        batch = list(filter(lambda x: x[0] is not None, batch))
        if len(batch) == 0:
            return torch.Tensor()
        return default_collate(batch)

    data_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npy')]
    label_df = pd.read_csv(os.path.join(data_dir, label_name),header=None)
    labels = label_df.values

    data_files_train, data_files_val, labels_train, labels_val = train_test_split(data_files, labels, test_size=val_size,
                                                                                  random_state=42)

    train_data = NewHeartbeatDataset(data_files_train, labels_train)
    val_data = NewHeartbeatDataset(data_files_val, labels_val)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=heartbeat_collate_fn)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=heartbeat_collate_fn)

    return train_loader,val_loader

def single_import(data_path,label_path,val_size,batch_size):
    with h5py.File(data_path, 'r') as hf:
        data = hf['data'][:]
    label = pd.read_csv(label_path)
    # Convert data to PyTorch tensors
    data_tensor = torch.tensor(data, dtype=torch.float32)
    label_tensor = torch.tensor(label.values, dtype=torch.float32)

    # Combine data and labels
    dataset = TensorDataset(data_tensor, label_tensor)

    # Split dataset
    train_dataset, test_dataset = train_test_split(dataset, test_size=val_size, random_state=42)

    # Create data loaders
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader,test_loader