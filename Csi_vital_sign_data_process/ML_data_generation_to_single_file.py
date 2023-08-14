# This file is used to process and generate a single file
# for machine learning. It can be used to save time for data reading
# when the amount of data is small. When the CSI sample rate is changed
# or when the number of selected sub-carriers is not the default value
# or when the number of selected angles is not the default value, please modify both lines of this file
# data = np.empty((0, 151, 10, 20))
# desired_length = 151
# in order to get the data in the correct format.
# The default step length of this programme is 15 seconds,
# and the step is 1 second. If you need to change this, please modify the code as needed.
import pandas as pd
import numpy as np
import h5py
from sklearn.model_selection import train_test_split
def ecg_get(ecg):
    mean_rr_ms = sum(ecg) / len(ecg)
    heart_rate = 60000 / mean_rr_ms
    return heart_rate
def single_file(ground_truth_file,h5_file,ML_data,ML_label):
    ecg = pd.read_csv(ground_truth_file)
    with h5py.File(h5_file, 'r') as f:
        matrix = np.array(f['matrix'])
        timestamps = np.array(f['timestamps'])
    def local_time(x):
        x = str(x)
        list = x.split(':')
        hours = int(list[0][-2:]) * 3600
        minutes = int(list[1]) * 60
        sec = float(list[2])
        return hours + minutes + sec

    ecg_time = ecg[['Phone timestamp']].applymap(local_time)
    csi_time = timestamps
    ecg_data = ecg[['RR-interval [ms]\n']]
    csi_data = matrix
    start_time = float(csi_time[0])
    end_time = float(csi_time[-1])
    time_gap = int(end_time-start_time)-15
    mask = ecg_time['Phone timestamp'].between(start_time,end_time)
    ecg_time = ecg_time[mask]
    ecg_data = ecg_data[mask]
    label = pd.DataFrame()
    data = np.empty((0, 151, 10, 20))
    desired_length = 151
    for i in range(time_gap):
        new_start_time = start_time+i
        new_end_time = new_start_time+15
        indices = np.where((timestamps >= new_start_time) & (timestamps <= new_end_time))
        data_csi = matrix[indices]


        if data_csi.shape[0] > desired_length:
            # 如果超过1501，删除多出的部分
            data_csi = data_csi[:desired_length]
        elif data_csi.shape[0] < desired_length:
            # 如果不足1501，用最后一行的数据补齐
            last_row = np.expand_dims(data_csi[-1], axis=0)
            num_rows_to_add = desired_length - data_csi.shape[0]
            rows_to_add = np.repeat(last_row, num_rows_to_add, axis=0)
            data_csi = np.concatenate((data_csi, rows_to_add), axis=0)


        data = np.concatenate((data, data_csi[np.newaxis, :]), axis=0)
        ecg_mask = ecg_time['Phone timestamp'].between(new_start_time,new_end_time)
        data_ecg =ecg_data[ecg_mask]
        kk = data_ecg['RR-interval [ms]\n'].values.tolist()
        heart = ecg_get(kk)
        df1 = pd.DataFrame([heart], columns=['heart rate'])
        label = pd.concat([label, df1])






    with h5py.File(ML_data, 'w') as hf:
        hf.create_dataset("data", data=data)
    label.to_csv(ML_label, index=False)


