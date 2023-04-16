import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import splrep, splev
from scipy.signal import detrend, find_peaks

file_name = "test2.csv" #记得改名
csvFile = open(file_name, 'w', newline='', encoding='utf-8')
writer = csv.writer(csvFile)
csvRow = []

f = open("Polar_H10_BF32822B_20230413_155357_ECG.txt", 'r', encoding='GB2312')
for line in f:
    csvRow = line.split(';')
    writer.writerow(csvRow)

f.close()
csvFile.close()

data = pd.read_csv(file_name)
ecg1 = np.array(pd.read_csv(file_name))
time = ecg1[:, 2]
ecg1 = ecg1[:, 3]
ecg = ecg1[408:]/1000#这里需要去掉一段时间的的数据，因为polar设备在前几秒的数据不可用，去除的数量为异常数量*2
#这个地方需要先plot一下全部的，寻找异常结束的时间
time = time[-len(ecg):]
# Load and preprocess data
#ecg = electrocardiogram()
sf_ori = 360
sf = 130
dsf = sf / sf_ori
#ecg = resample(ecg, dsf)

window = (time[-2]-time[0])/1000
start =15 #这里的设置会影响结果，需要看一下
# R-R peaks detection
rr, _ = find_peaks(ecg, distance=40, height=0.5)

plt.plot(ecg)
plt.plot(rr, ecg[rr], 'o')
plt.title('ECG signal')
plt.xlabel('Samples')
_ =plt.ylabel('Voltage')
plt.show()
# R-R interval in ms
rr = (rr / sf) * 1000
rri = np.diff(rr)


# Interpolate and compute HR
def interp_cubic_spline(rri, sf_up=2):
    """
    Interpolate R-R intervals using cubic spline.
    Taken from the `hrv` python package by Rhenan Bartels.

    Parameters
    ----------
    rri : np.array
        R-R peak interval (in ms)
    sf_up : float
        Upsampling frequency.

    Returns
    -------
    rri_interp : np.array
        Upsampled/interpolated R-R peak interval array
    """
    rri_time = np.cumsum(rri) / 1000.0
    time_rri = rri_time - rri_time[0]
    time_rri_interp = np.arange(0, time_rri[-1], 1 / float(sf_up))
    tck = splrep(time_rri, rri, s=0)
    rri_interp = splev(time_rri_interp, tck, der=0)
    return rri_interp


sf_up = 2
rri_interp = interp_cubic_spline(rri, sf_up)
hr = 1000 * (60 / rri_interp)
print('Mean HR: %.2f bpm' % np.mean(hr))

# Detrend and normalize
edr = detrend(hr)
edr = (edr - edr.mean()) / edr.std()

# Find respiratory peaks
resp_peaks, _ = find_peaks(edr, height=0, distance=sf_up)

# Convert to seconds
resp_peaks = resp_peaks
resp_peaks_diff = np.diff(resp_peaks) / sf_up

# Plot the EDR waveform
plt.plot(edr, '-')
plt.plot(resp_peaks, edr[resp_peaks], 'o')
_ = plt.title('ECG derived respiration')
plt.show()
# Extract the mean respiratory rate over the selected window
mresprate = resp_peaks.size / window
print('Mean respiratory rate: %.2f Hz' % mresprate)
print('Mean respiratory period: %.2f seconds' % (1 / mresprate))
print('Respiration RMS: %.2f seconds' % np.sqrt(np.mean(resp_peaks_diff**2)))
print('Respiration STD: %.2f seconds' % np.std(resp_peaks_diff))