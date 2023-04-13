clc
% This .m file allows you change the PicoScenes file to .mat file, so that
% you do not need the tooldbox to process the data. 
%To run this code, you have to install PicoScenes Toolbox, please find the
%code on https://ps.zpj.io/matlab.html

saved_filename = "rx_2_230329_140123.mat";
csi_data = rx_2_230329_140123; % change the file name, double click the file before run the code
save(saved_filename,"csi_data")

