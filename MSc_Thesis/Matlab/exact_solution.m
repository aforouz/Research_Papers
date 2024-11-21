function [value] = exact_solution(p, t)
x = p(1);
y = p(2);

%% Power 2
value = power(x * (1 - x), 2) * power(y * (1 - y), 2) * exp(-t) * 10;
