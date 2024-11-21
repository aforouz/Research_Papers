function [model] = set_model_data(model_config, M_x, M_y, N_t)

%% Seprate data
model.p_beta = model_config.p_beta;
model.p_gamma = model_config.p_gamma;
model.K_x = model_config.K_x;
model.K_y = model_config.K_y;
model.T_eps = model_config.T_eps;

%% Step
model.M_beta = M_x;
model.M_gamma = M_y;
model.N_time = N_t;