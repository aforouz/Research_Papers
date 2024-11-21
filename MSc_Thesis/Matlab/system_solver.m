function [model] = system_solver(model)

tau = model.tau/2;

%% Matrix




%% Time
U = model.psi;

disp("system solver : Iteration");
tic;
% disp("----------------------------------------");
for it = 1:model.N_time
    % disp("----- iteration");
    % disp(it);

    PSI_h = model.mass_matrix + (tau(it) * model.stiff_matrix);
    PSI_n = model.mass_matrix - (tau(it) * model.stiff_matrix);
    % PSI = PSI_h \ PSI_n;
    % F =  model.load_vector(:, it);

    F = (PSI_n*U + model.load_vector(:, it));
    U = PSI_h \ F;
end

model.timer = toc;
%% End
model.numerical_solution = U;