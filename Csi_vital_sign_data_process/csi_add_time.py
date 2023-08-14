import numpy as np
import h5py
import pandas as pd
from datetime import datetime, timedelta
def csi_add_time(end_time_str,file_name):
    with h5py.File(file_name, 'r') as f:
        matrix = f['matrix'][:]
    interval = timedelta(milliseconds=100).total_seconds()
    total_timestamps = matrix.shape[0]
    end_time_parts = end_time_str.split(':')
    end_time = int(end_time_parts[0]) * 3600 + int(end_time_parts[1]) * 60 + float(end_time_parts[2])  # 转换为秒

    timestamps = np.array([end_time - i * interval for i in range(total_timestamps)])
    timestamps = np.flip(timestamps)

    with h5py.File(file_name, 'w') as f:
        f.create_dataset('matrix', data=matrix)
        f.create_dataset('timestamps', data=timestamps)