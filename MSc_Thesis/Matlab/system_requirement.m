function [model] = system_requirement(model)

%% Initial
disp("system requirement : initial condition");
tic;

model.psi = zeros(model.K_h, 1);
for iKh = 1:model.K_h
    model.psi(iKh) = exact_solution(model.node(:, iKh), model.t_start);
end

toc;

%% Bound
disp("system requirement : boundary condition");
tic;

model.bound_value = zeros(length(model.bound_point), model.N_time+1);
for ib = 1:length(model.bound_point)
    for it = 1:model.N_time+1
        model.bound_value(ib, it) = exact_solution(model.bound_point(:, ib), model.t(it)); % need to first step of time
    end
end

toc;

%% M_h
h_x = model.h_x;
h_y = model.h_y;
M_beta = model.M_beta;
M_gamma = model.M_gamma;

M_coeff = h_x * h_y / 12;

disp("system requirement : mass matrix");
tic;

M_1 = spdiags(M_coeff * [1, 6, 1], -1:1, M_beta - 1, M_beta - 1);

M_2 = spdiags(M_coeff * [1, 1] , 0:1, M_beta - 1, M_beta - 1);

model.mass_matrix = kron(speye(M_gamma - 1), M_1) + ...
    kron(spdiags(1, 1, M_gamma - 1, M_gamma - 1), M_2) + ...
    kron(spdiags(1, -1, M_gamma - 1, M_gamma - 1), transpose(M_2));

toc;

%% A_h
p_beta = model.p_beta;
p_gamma = model.p_gamma;

%% A_x_beta
disp("system requirement : stiff matrix beta");
tic;

A_1_beta = create_A_1(p_beta, M_beta);
A_2_beta = create_A_2(p_beta, M_beta);

A_beta = kron(speye(M_gamma - 1), A_1_beta) + kron(spdiags(1, 1, M_gamma - 1, M_gamma - 1), A_2_beta) + ...
    kron(spdiags(1, -1, M_gamma - 1, M_gamma - 1), transpose(A_2_beta));

A_x_beta = power(h_x, 1 - 2*p_beta) * h_y / (2 * cos(p_beta * pi) * gamma(5 - 2*p_beta)) * A_beta;

toc;

%% A_y_gamma
disp("system requirement : stiff matrix beta");
tic;

A_1_gamma = create_A_1(p_gamma, M_gamma);
A_2_gamma = create_A_2(p_gamma, M_gamma);

A_gamma = kron(speye(M_beta - 1), A_1_gamma) + kron(spdiags(1, 1, M_beta - 1, M_beta - 1), A_2_gamma) + ...
    kron(spdiags(1, -1, M_beta - 1, M_beta - 1), transpose(A_2_gamma));

P_pi = create_permutation_matrix(M_beta, M_gamma);

A_y_gamma = h_x * power(h_y, 1 - 2*p_gamma) / (2 * cos(p_gamma * pi) * gamma(5 - 2*p_gamma)) * transpose(P_pi) * A_gamma * P_pi;

toc;

%% Sum
model.stiff_matrix = model.K_x*A_x_beta + model.K_y*A_y_gamma;

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Help function : create A_2_rho

function [A] = create_A_1(rho, M_rho) 
A = zeros(M_rho - 1);

for i = 1:M_rho-1
    A(i, i) = power(2, 6 - 2*rho) + 16*rho - 40;
end

for j = 1:M_rho-2
    A(j, j + 1) = 2 * power(3, 4 - 2*rho) + ...
        (2*rho - 6) * power(2, 5 - 2*rho) - 16*rho + 34;
    A(j + 1, j) = A(j, j + 1);
end

for l = 2:M_rho-2
    for k = 1:(M_rho - l - 1)
        A(k, k + l) = 4 * (4 - 2*rho) * (-power(l - 1, 3 - 2*rho) + 2*power(l, 3 - 2*rho) - power(l+1, 3 - 2*rho)) ...
            - 2*power(l - 2, 4 - 2*rho) + 4*power(l - 1, 4 - 2*rho) ...
            - 4*power(l + 1, 4 - 2*rho) + 2*power(l + 2, 4 - 2*rho);
        A(k + l, k) = A(k, k + l);
    end
end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Help function : create A_1_rho

function [A] = create_A_2(rho, M_rho) 
A = zeros(M_rho - 1);

for i = 1:M_rho-1
    A(i, i) = 4 - power(2, 4 - 2*rho)*rho;
end

for j = 1:M_rho-2
    A(j, j + 1) = 4 - power(2, 4 - 2*rho)*rho;
end

for l = 2:M_rho-2
    for k = 1:(M_rho - l - 1)
        A(k, k + l) = (4 - 2*rho) * (power(l - 2, 3 - 2*rho) - power(l - 1, 3 - 2*rho) - power(l, 3 - 2*rho) + power(l+1, 3-2*rho)) ...
            + 2*power(l - 2, 4 - 2*rho) - 6*power(l - 1, 4 - 2*rho) ...
            + 6*power(l, 4 - 2*rho) - 2*power(l + 1, 4 - 2*rho);
    end
end

for m = 1:M_rho-2
    for p = 1:(M_rho - m - 1)
        A(p + m, p) = (4 - 2*rho) * (power(m - 1, 3 - 2*rho) - power(m, 3 - 2*rho) - power(m + 1, 3 - 2*rho) + power(m + 2, 3 - 2*rho)) ...
            + 2*power(m - 1, 4 - 2*rho) - 6*power(m, 4 - 2*rho) ...
            + 6*power(m + 1, 4 - 2*rho) - 2*power(m + 2, 4 - 2*rho);
    end
end

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Help function : Permutation matrix
function [P] = create_permutation_matrix(M_beta, M_gamma)
M_beta_gamma = (M_beta - 1) * (M_gamma - 1);
P = sparse(M_beta_gamma);

indexTemp = transpose(1:(M_beta-1):(1+(M_beta-1)*(M_gamma-2)));
index = indexTemp;
for i = 1:M_beta-2
    index(:, end+1) = indexTemp + i;
end
index = index(:);

for m = 1:M_beta_gamma
    P(index(m), m) = 1;
end