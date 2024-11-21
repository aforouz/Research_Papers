%% Start
clc;
clear;
close all;
format long;

%% Order
config_data = scanf_config_data();

%% Run
for dxyt = 1:length(config_data.order_config.Dt)
    disp("-------------------------------------------------- + --------------------------------------------------");
    disp("Time: ");
    disp(power(config_data.order_config.Bt, config_data.order_config.Dt(dxyt)));
    disp("Space: ");
    disp(power(config_data.order_config.Bxy, config_data.order_config.Dxy(dxyt)));
    disp("-------------------------------------------------- + --------------------------------------------------");

    %% Problem
    model = set_model_data(config_data.model_config,...
            power(config_data.order_config.Bxy, config_data.order_config.Dxy(dxyt)),...
            power(config_data.order_config.Bxy, config_data.order_config.Dxy(dxyt)),...
            power(config_data.order_config.Bt, config_data.order_config.Dt(dxyt)));        

    model = initialize_model(model);
    
    %% Mesh
    model = mesh_generator(model);

    %% Requirement
    model = system_requirement(model);
    model = right_side_space(model);
    model = right_side_time(model);

    %% Solver
    model = system_solver(model);
    config_data.order_config.time_step(dxyt) = model.timer;
    
    %% Error
    config_data.order_config.error_step(dxyt) = print_result(model, config_data.print_config);
end

print_order(config_data);