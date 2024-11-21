function [value] = source_term_space(p_beta, p_gamma, K_x, K_y, p)
x = p(1);
y = p(2);

%% u_t = X * Y * T'
valueUxy = power(x * (1 - x), 2) * power(y * (1 - y), 2);

%% u_xx + u_yy = -(K_x*X''*Y + K_y*X*Y'') * T

valueDxy = ...
K_x * power(y - power(y, 2), 2) / cos(p_beta * pi) * ...
( ...
(power(x, 2 - 2*p_beta) + power(1 - x, 2 - 2*p_beta)) / gamma(3 - 2*p_beta) ...
-6 * (power(x, 3 - 2*p_beta) + power(1 - x, 3 - 2*p_beta)) / gamma(4 - 2*p_beta) ...
+12 * (power(x, 4 - 2*p_beta) + power(1 - x, 4 - 2*p_beta)) / gamma(5 - 2*p_beta) ...
) ...
    + ...
K_y * power(x - power(x, 2), 2) / cos(p_gamma * pi) * ...
( ...
(power(y, 2 - 2*p_gamma) + power(1 - y, 2 - 2*p_gamma)) / gamma(3 - 2*p_gamma) ...
-6 * (power(y, 3 - 2*p_gamma) + power(1 - y, 3 - 2*p_gamma)) / gamma(4 - 2*p_gamma) ...
+12 * (power(y, 4 - 2*p_gamma) + power(1 - y, 4 - 2*p_gamma)) / gamma(5 - 2*p_gamma) ...
) ;

%% Result
value = [valueUxy, valueDxy];