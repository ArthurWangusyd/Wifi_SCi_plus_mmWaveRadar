function Farsenes_getdata_v2(filepath,savepath,sampling_rate,n_sub,n_angle)

    csi_data = opencsi(filepath); %load the data
    len = length(csi_data);
        if len >1
            first = zeros(57, len);
            second = zeros(57, len);
            count = 1;
    
            for i = 1:len
                csi = csi_data{i,1}.CSI.CSI;
        
                if all(size(csi) == [57, 1, 2])
                    first(:, count) = csi(:, 1, 1);
                    second(:, count) = csi(:, 1, 2);
                    count = count + 1;
                end
            end
    
    % Trim arrays to actual size
    first = first(:, 1:count-1);
    second = second(:, 1:count-1);
        else
            csi_data = csi_data{1,1}.CSI.CSI;
            first = csi_data(:,1:57).';
            second = csi_data(:,58:114).';
        end
    
    ratio = first./second;
    
    
    
    % Seting data
    theta = 0:pi/50:2*pi-pi/50; % Projection angle
    
    % Savitzky-Golay Filter to smoothen the data
    farmesize = 103;
    polynomialOrder = 4;
    ratio_filter = sgolayfilt(ratio', polynomialOrder, farmesize)';
    
    % The ratio of maximal BNR to the sum of all BNRs (weights)
    bnr_values = zeros(size(ratio,1), size(ratio, 1));
    avg_weights = zeros(size(ratio,1),1);
    sorted_angle = zeros(size(ratio,1),n_angle);
    for sub_num = 1:size(ratio,1)
        % Projection of the data
        sub = ratio_filter(sub_num,:);
        rotMatrix = [cos(theta); sin(theta)].';
        csiMatrix = [real(sub); imag(sub)];
        matrix = rotMatrix * csiMatrix;
        
        % Calculate BNR for each window
        fft_size = 1024;
        freq_range = [60/60, 120/60];
        sampling_rate = double(sampling_rate);
        bin_range = round(freq_range * fft_size / sampling_rate) + 1;
        for i = 1:size(matrix, 1)
            window = matrix(i, :);
            window_padded = [window, zeros(1, fft_size-length(window))];
            fft_window = abs(fft(window_padded));
            bnr_values(sub_num,i) = max(fft_window(bin_range(1):bin_range(2))) / sum(fft_window);
        end
    
        % Store average weights of each subcarries
        [sorted_bnr_values, sorted_bnr_indices] = sort(bnr_values,2,'descend');
        avg_weights(sub_num,1) = sum(sorted_bnr_values(sub_num,1:n_angle))/sum(bnr_values(sub_num,:));
    end
    
    % Store angles of each subcarries
    for sub_num = 1:size(ratio,1)
        sorted_angle(sub_num,:) = theta(sorted_bnr_indices(sub_num,1:n_angle));
    end
    
    % Get the n_sub subcarrier with the largest mean value
    [sorted_avgw_values, sorted_avgw_indices] = sort(avg_weights,'descend'); 
    top_n_max_avgw_value = sorted_avgw_values(1:n_sub);
    top_n_max_avgw_indices = sorted_avgw_indices(1:n_sub);
    
    % Obtaining target-specific values
    matrix_measurements = zeros(size(top_n_max_avgw_indices,1),n_angle,size(ratio,2));
    i_matrix_measurements = 0;
    for i_n_max_avg = 1:size(top_n_max_avgw_indices,1)
        maxavg_sub_indices = top_n_max_avgw_indices(i_n_max_avg);
        i_matrix_measurements = i_matrix_measurements+1;
        % Projection of the data
        sub = ratio_filter(maxavg_sub_indices,:);
        rotMatrix = [cos(sorted_angle(maxavg_sub_indices,:)); sin(sorted_angle(maxavg_sub_indices,:))].';
        csiMatrix = [real(sub); imag(sub)];
        matrix_measurements_2d = rotMatrix * csiMatrix;
        matrix_measurements(i_matrix_measurements,:,:) = matrix_measurements_2d;
    end
    
    data_to_save = matrix_measurements;
    if exist(savepath, 'file')
        delete(savepath)
    end
    
    h5create(savepath,'/matrix', size(data_to_save), 'Datatype', 'double');
    h5write(savepath, '/matrix', data_to_save)
end

