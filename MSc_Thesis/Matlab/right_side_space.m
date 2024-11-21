function [model] = right_side_space(model)

%% Value of problem : source term space integral
point = model.integral_2D(1).point;
weight = model.integral_2D(1).weight;

p_beta = model.p_beta;
p_gamma = model.p_gamma;
K_x = model.K_x;
K_y = model.K_y;
J = model.h_x*model.h_y;

model.load_vector = zeros(model.K_h, 2);

disp("system requirement : load vector");
tic;

for iKh = 1:model.K_h
    value = [0, 0];

    p0 = model.node(:, iKh);
    for iTh = 1:model.T_h
        p21 = model.element(iTh).p2 - model.element(iTh).p1;
        p31 = model.element(iTh).p3 - model.element(iTh).p1;

        xyst = @(st)(p0 + model.element(iTh).p1 + [p21(1), p31(1); p21(2), p31(2)]*st);

        if iTh < 3
            phi = @(st)(st(2));
        elseif iTh < 5
            phi = @(st)(1-st(1)-st(2));
        else
            phi = @(st)(st(1));
        end

        f_phi = @(st)(source_term_space(p_beta, p_gamma, K_x, K_y, xyst(st)) * phi(st));

        element_value = 0;
        for ip = 1:length(weight)
            element_value = element_value + weight(ip)*f_phi(point(:, ip));
        end
        value = value + element_value;
    end
    
    model.load_vector(iKh, :) = J*value;
end

toc;