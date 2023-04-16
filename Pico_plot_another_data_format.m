%csi文件有时会有两种格式，一种是2*57，一种是1*114.这种格式的数据将两个天线接收的数据放在一起，前57个是第一个天线，后57个是第二个天线
clc
csi_data = rx_2_230329_151402{1,1}.CSI.CSI; %load the data
len = length(csi_data);
rx1_csi = csi_data(:,1:57).';
rx2_csi = csi_data(:,58:114).';
figure;
subplot(1,2,1);
plot(abs(rx1_csi));
subplot(1,2,2);
plot(angle(rx1_csi));
title("before calibration")

SubcarrierIndex = (-28:28)';
csi_all_calibrated = rx1_csi;
for i = 2:size(rx1_csi, 2)
    csi_curr = csi_all_calibrated(:, i);
    csi_prev = csi_all_calibrated(:, i-1);
    % warning("I am using the first CSI as reference to avoid accumulating error, to verify that the plotted curves will overlap. After improving the code with fminunc, we can use the preceding CSI as reference.")
        
    % Find alpha, beta, theta such that
    % | csi_curr .* exp(1i * (alpha + SubcarrierIndex / 64 * theta)) * beta
    % -  csi_prev |   is minimized
    % or, in an equivalent formulation
    % minimize | csi_curr * (a + 1i * b) .* exp(1i * SubcarrierIndex / 64 * theta) 
    % -  csi_prev |   
    
    csi_curr_cali = calibrate_csi(csi_curr, csi_prev);
    
%     figure;
%     subplot(1,2,1);
%     plot(abs(csi_curr_cali), 'r');
%     hold on;
%     plot(abs(csi_prev), 'b--');
%     legend('calibrated', 'ref');
%     subplot(1,2,2);
%     plot(angle(csi_curr_cali), 'r');
%     hold on;
%     plot(angle(csi_prev), 'b--');
%     legend('calibrated', 'ref');
    
    csi_all_calibrated(:, i) = csi_curr_cali;
end

figure;
subplot(1,2,1);
plot(abs(csi_all_calibrated(:, :)));
subplot(1,2,2);
plot(angle(csi_all_calibrated(:, :)));
title("After calibration")    