function csi_calibrated = calibrate_csi(csi_input, csi_ref)
% match csi_input again csi_ref

t_delay = estimate_excess_delay_20M(csi_ref ./ csi_input);
csi_input_1 = csi_input .* exp(1i * 2 * pi * 312.5e3 * (-28:28)' * t_delay);


c = csi_input_1'*csi_ref / (csi_input_1'*csi_input_1);
csi_calibrated = csi_input_1 * c;

% figure;
% plot(unwrap(angle(csi_input)));
% hold on;
% plot(unwrap(angle(csi_ref)));
% plot(unwrap(angle(csi_calibrated)), '--');
% legend('input', 'ref', 'calibrated')

% In the context of 
% minimize | csi_curr * (a + 1i * b) .* exp(1i * SubcarrierIndex / 64 * theta) 
% -  csi_prev | 
%  theta = 2 * pi * 312.5e3 * t_delay * 64
%  c = a + 1i * b
theta = 2 * pi * 312.5e3 * t_delay;
x0 = [real(c); imag(c); theta];

obj_func = @(x)sum(abs(csi_input .* exp(1i * (-28:28)' * x(3)) * (x(1) + 1i*x(2)) - csi_ref).^2);
x = fminunc(obj_func, x0);
% 
% disp(["loss before refinement ", num2str(obj_func(x0))])
% disp(["loss after refinement ", num2str(obj_func(x))])

csi_calibrated = csi_input .* exp(1i * (-28:28)' * x(3)) * (x(1) + 1i*x(2));
 
end



function t_delay = estimate_excess_delay_20M(c)
% reusing past code, here we return values in seconds
assert(length(c) == 57);
fft_size = 64 * 64;
ts = 1 / (fft_size * 312.5e3);
c_windowed = c(:) .* hamming(57);
c_padded = [c_windowed; zeros(fft_size - 57, 1)];
x = ifftshift(abs(ifft(c_padded)));
[~, i_max] = max(x);
t_delay = ts * (fft_size / 2 - i_max);
end



