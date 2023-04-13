clc

csi_data = load('csi.mat').rx_2_230329_151213; %load the data
%If you have the picoscenes toolbox, you can change the code to csi_data =
%filename(without.csv)
len = length(csi_data); 

first_phase = [];
first_mag = []; 
second_phase = [];
second_mag = []; %phase and amplitude for two antenna
first_csi= [];
second_csi = []; %csi data(complex number)
for i = 1:len
    csi = squeeze(csi_data{i,1}.CSI.CSI).';
    csi_phase = squeeze(csi_data{i,1}.CSI.Phase).';
    csi_mag= squeeze(csi_data{i,1}.CSI.Mag).';
    if size(csi_phase) == [2,57]
    %The csi should be 2*57, however, there is some empty data,remove them
        first_phase = [first_phase;csi_phase(1,:)];
        second_phase= [second_phase;csi_phase(2,:)];
        first_mag = [first_mag;csi_mag(1,:)];
        second_mag= [second_mag;csi_mag(2,:)];
        first_csi = [first_csi;csi(1,:)];
        second_csi = [second_csi:csi(2,:)];
    end
       
   
end
figure;
subplot(1,2,1);
plot(abs(first_csi).');
subplot(1,2,2);
plot(angle(first_csi).');
title("before calibration") %plot the figure 
