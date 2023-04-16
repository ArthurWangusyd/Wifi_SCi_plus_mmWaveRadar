% clc
% 
% csi_data = load('csi.mat').rx_2_230329_151213; %load the data
% len = length(csi_data);
% 
% first_phase = [];
% first_mag = []; 
% second_phase = [];
% second_mag = []; %phase and amplitude
% first_csi= [];
% second_csi = []; %csi data(complex number)
% for i = 1:len
%     csi = squeeze(csi_data{i,1}.CSI.CSI).';
%     csi_phase = squeeze(csi_data{i,1}.CSI.Phase).';
%     csi_mag= squeeze(csi_data{i,1}.CSI.Mag).';
%     if size(csi_phase) == [2,57]
%     %csi格式是2*57，有一些空包
%         first_phase = [first_phase;csi_phase(1,:)];
%         second_phase= [second_phase;csi_phase(2,:)];
%         first_mag = [first_mag;csi_mag(1,:)];
%         second_mag= [second_mag;csi_mag(2,:)];
%         first_csi = [first_csi;csi(1,:)];
%         second_csi = [second_csi:csi(2,:)];
%     end
%        
%    
% end

% A = first_phase.';
% plot(A)% plot the phase of the first antenna


clc;


csi_data = load('csi.mat').rx_2_230329_151213; %load the data
len = length(csi_data);

antenna_idx = 1;
csi_all = [];
for i = 1:len
    csi = csi_data{i,1}.CSI.CSI;
    if ~all(size(csi) == [57, 1, 2])
        continue
    end
    
    c = csi(:, 1, antenna_idx);
    csi_all = [csi_all, c];
end

figure;
subplot(1,2,1);
plot(abs(csi_all));
subplot(1,2,2);
plot(angle(csi_all));
title("before calibration")


SubcarrierIndex = (-28:28)';
csi_all_calibrated = csi_all;
for i = 2:size(csi_all, 2)
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