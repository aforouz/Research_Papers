function [norm_error] = print_result(model, print_config)

%% Value of problem : source term
norm_error = norm_L(model);

%% Print error
if print_config.print_error
    file_ID = fopen(print_config.address + "norm_error/norm_error_x_" + model.M_beta + "_y_" + model.M_gamma + "_t_" + model.N_time + ".txt", "w");
    fprintf(file_ID, "norm error: %E\n", norm_error);
    fclose(file_ID);
end

%% Print plot
if print_config.print_plot
    
    %% Value of exact solution
    exact_value = zeros(model.K_h, 1);
    for iKh = 1:model.K_h
        exact_value(iKh) = exact_solution(model.node(:, iKh), model.t_end);
    end

    % Boundary
    uE = zeros(model.M_beta + 1, model.M_gamma + 1);
    uE(:, 1) = model.bound_value(1:(model.M_beta + 1), end);
    uE(:, end) = model.bound_value((end-model.M_beta):end, end);
    
    bound_index = model.M_beta + 2;
    for ig = 2:model.M_gamma
        uE(1, ig) = model.bound_value(bound_index, end);
        uE(end, ig) = model.bound_value(bound_index + 1, end);
        bound_index = bound_index + 2;
    end
    
    % Value
    uN = uE;
    uE(2:end-1, 2:end-1) = reshape(exact_value, model.M_beta - 1, model.M_gamma - 1);
    uN(2:end-1, 2:end-1) = reshape(model.numerical_solution, model.M_beta - 1, model.M_gamma - 1);
    
    % Mesh
    x = linspace(model.x_start, model.x_end, model.M_beta + 1);
    y = linspace(model.y_start, model.y_end, model.M_gamma + 1);
    [X, Y] = meshgrid(x, y);
    X = transpose(X);
    Y = transpose(Y);

    % Plot
    fig = figure('Visible','on');
    subplot(1, 2, 1);
    surf(X, Y, uE);
    title("Exact");
    subplot(1, 2, 2);
    surf(X, Y, uN);
    title("Numeric");
    grid on;
    saveas(fig, print_config.address + "plot/plot_x_" + model.M_beta + "_y_" + model.M_gamma + "_t_" + model.N_time);
    close(fig);
end