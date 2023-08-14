# This file is used to process and generate multi files
# for machine learning. Use this method when the size of your data is large.When the CSI sample rate is changed
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
import os

def local_time(x):
    x = str(x)
    list = x.split(':')
    hours = int(list[0][-2:]) * 3600
    minutes = int(list[1]) * 60
    sec = float(list[2])
    return hours + minutes + sec
def ecg_get(ecg):
    mean_rr_ms = sum(ecg) / len(ecg)
    heart_rate = 60000 / mean_rr_ms
    return heart_rate
def multi_files(ground_truth_file,h5_file,save_path,ML_label):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    ecg = pd.read_csv(ground_truth_file)
    with h5py.File(h5_file, 'r') as f:
        matrix = np.array(f['matrix'])
        timestamps = np.array(f['timestamps'])

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
    try:
        full_path = os.path.join(save_path, ML_label)
        start_idx = pd.read_csv(full_path).shape[0]+1
    except:
        start_idx = 0
    print(start_idx)
    for i in range(time_gap):
        new_start_time = start_time+i
        new_end_time = new_start_time+15
        indices = np.where((timestamps >= new_start_time) & (timestamps <= new_end_time))
        data_csi = matrix[indices]
        desired_length = 151
        if data_csi.shape[0] > desired_length:
            # 如果超过151，删除多出的部分
            data_csi = data_csi[:desired_length]
        elif data_csi.shape[0] < desired_length:
            # 如果不足151，用最后一行的数据补齐
            last_row = np.expand_dims(data_csi[-1], axis=0)
            num_rows_to_add = desired_length - data_csi.shape[0]
            rows_to_add = np.repeat(last_row, num_rows_to_add, axis=0)
            data_csi = np.concatenate((data_csi, rows_to_add), axis=0)
        np.save(os.path.join(save_path, 'data_{}.npy'.format(start_idx + i)), data_csi)
        ecg_mask = ecg_time['Phone timestamp'].between(new_start_time,new_end_time)
        data_ecg =ecg_data[ecg_mask]
        kk = data_ecg['RR-interval [ms]\n'].values.tolist()
        heart = ecg_get(kk)
        df1 = pd.DataFrame([heart], columns=['heart rate'])
        label= pd.concat([label, df1])

    label.to_csv(full_path, mode='a', header=False, index=False)