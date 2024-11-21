function [model] = right_side_time(model)

%% Value of problem : source term time integral
point = model.integral_1D(2).point;
weight = model.integral_1D(2).weight;

t = model.t;
tau = model.tau;

load_vector_time = zeros(2, model.N_time); % no need to first step of time in load vector

disp("system requirement : time of load vector");
tic;

% tn-1 ... tn
for n = 1:model.N_time
    f_L = @(ts)(source_term_time(tau(n)*ts + t(n)));
    
    value = [0; 0];
    for ip = 1:length(weight)
        value = value + weight(ip)*f_L(point(ip));
    end
    load_vector_time(:, n) = tau(n)*value;
end

model.load_vector = model.load_vector*load_vector_time;

toc;