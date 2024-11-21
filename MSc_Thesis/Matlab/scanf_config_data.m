function [config_data] = scanf_config_data()

%% Read config file
temp_data = readmatrix("config_data.csv");

%% Seprate data
config_data.model_config.p_beta = temp_data(1, 2);
config_data.model_config.p_gamma = temp_data(2, 2);
config_data.model_config.K_x = temp_data(3, 2);
config_data.model_config.K_y = temp_data(4, 2);
config_data.model_config.T_eps = temp_data(5, 2);

config_data.order_config.Bt = temp_data(6, 2);
config_data.order_config.Dt = temp_data(7, 2):temp_data(8, 2);
config_data.order_config.Bxy = temp_data(9, 2);
config_data.order_config.Dxy = temp_data(10, 2):temp_data(11, 2);

config_data.print_config.print_error = temp_data(12, 2);
config_data.print_config.print_plot = temp_data(13, 2);

%% Error result
config_data.order_config.error_step = zeros(length(config_data.order_config.Dxy), 1);
config_data.order_config.time_step = zeros(length(config_data.order_config.Dxy), 1);

%% Output
mkdir("output");
config_data.print_config.address = "output/result_eps_" + config_data.model_config.T_eps + ...
    "_b_" + config_data.model_config.p_beta + "_g_" + config_data.model_config.p_gamma + ...
    "_Kx_" + config_data.model_config.K_x + "_Ky_" + config_data.model_config.K_y + ...
    "_t_"  + config_data.order_config.Bt + "_" + config_data.order_config.Dt(1) + "_" + config_data.order_config.Dt(end) + ...
    "_x_y_" + config_data.order_config.Bxy + "_" + config_data.order_config.Dxy(1) + "_" + config_data.order_config.Dxy(end) + "/";

mkdir(config_data.print_config.address);

if config_data.print_config.print_error
    mkdir(config_data.print_config.address + "norm_error");
end

if config_data.print_config.print_plot
    mkdir(config_data.print_config.address + "plot");
end