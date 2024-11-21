function [value] = source_term_time(t)

%% u_t = X * Y * T'
valueDt = -exp(-t) * 10;

%% u_xx + u_yy = -(K_x*X''*Y + K_y*X*Y'') * T
valueUt = exp(-t) * 10;

%% Result
value = [valueDt; valueUt];