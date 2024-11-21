function [norm_error] = norm_L(model)

%% Value of problem : source term space integral
point = model.integral_2D(1).point;
weight = model.integral_2D(1).weight;
h_x = model.h_x;
h_y = model.h_y;
J = h_x*h_y;

% Boundary
uE = zeros(model.M_beta + 1, model.M_gamma + 1);
uE(2:end-1, 2:end-1) = reshape(model.numerical_solution, model.M_beta - 1, model.M_gamma - 1);
uE = uE(:);

%% all points
x = linspace(model.x_start, model.x_end, model.M_beta + 1);
y = linspace(model.y_start, model.y_end, model.M_gamma + 1);
node = [kron(ones(1, model.M_gamma + 1), x); kron(y, ones(1, model.M_beta + 1))];

%% all elements
T_h = 2 * model.M_beta * model.M_gamma;
element = zeros(3, T_h);

index_diff = model.M_beta + 1 + 1;
down_index = 1;
up_index = 1 + index_diff;

for iTh = 1:2:T_h
    element(:, iTh) = [up_index - index_diff; up_index - 1; up_index];
    element(:, iTh+1) = [down_index; down_index + 1; down_index + index_diff];

    down_index = down_index + 1;
    up_index = up_index + 1;

    if rem(down_index, model.M_beta + 1) == 0
        down_index = down_index + 1;
        up_index = up_index + 1;
    end
end

%% Norm Value
error_vector = zeros(T_h, 1);
for iTh = 1:T_h
    p1 = node(:, element(1, iTh));
    p2 = node(:, element(2, iTh));
    p3 = node(:, element(3, iTh));
    u1 = uE(element(1, iTh));
    u2 = uE(element(2, iTh));
    u3 = uE(element(3, iTh));
    
    x1 = p1(1);
    y1 = p1(2);
    if rem(iTh, 2)==1
        phi = @(xy) (u1 + ((u2 - u3)*x1)/h_x + ((u1 - u2)*y1)/h_y + (-u2 + u3)/h_x*xy(1) + (-u1 + u2)/h_y*xy(2));
    else
        phi = @(xy) (u1 + ((u1 - u2)*x1)/h_x + ((u2 - u3)*y1)/h_y + (-u1 + u2)/h_x*xy(1) + (-u2 + u3)/h_y*xy(2));
    end
    xyst = @(st)(p2 + [p3(1) - p2(1), p1(1) - p2(1); p3(2) - p2(2), p1(2) - p2(2)]*st);

    % gauss integral
    u_uht = @(st)(power(exact_solution(xyst(st), model.t_end) - phi(xyst(st)), 2));

    value = 0;
    for ip = 1:length(weight)
        value = value + weight(ip)*u_uht(point(:, ip));
    end

    error_vector(iTh) = J*value;
end

%% Error
norm_error = sqrt(sum(error_vector));