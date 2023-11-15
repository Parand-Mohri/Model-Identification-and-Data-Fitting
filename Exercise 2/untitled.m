% Parameters
order = 3;  % ARX model order
num_measurements = 1000;

% True system coefficients
a = [-0.5; -0.6; -0.3];
b = [0.2; 1.4; -0.5];

% Simulation of the ARX system
u = randn(num_measurements, 1);  % Input sequence (white noise)
y = zeros(num_measurements, 1);  % Output sequence

for t = order+1:num_measurements
    y(t) = -a.' * y(t-order:t-1) + b.' * u(t-order:t-1) + randn();  % ARX model
end

% Recursive Least Squares Algorithm
theta_ls = zeros(2 * order, 1);  % Initialization
H = eye(2 * order);  % Initialization
L = 0;  % Initialization

parameter_estimates = zeros(num_measurements, 2 * order);

for N = order+1:num_measurements
    phi_N = [-y(N-1:-1:N-order); u(N-1:-1:N-order)];  % Regression vector
    
    % Apply lsrecursive function
    [theta_ls, H, L] = lsrecursive(theta_ls, H, L, phi_N, y(N));
    
    parameter_estimates(N, :) = theta_ls;
end

% Plot parameter estimates
figure;
subplot(2, 1, 1);
plot(1:num_measurements, parameter_estimates(:, 1:order), 'LineWidth', 2);
title('Estimated Coefficients a');
legend('a_1', 'a_2', 'a_3');

subplot(2, 1, 2);
plot(1:num_measurements, parameter_estimates(:, order+1:end), 'LineWidth', 2);
title('Estimated Coefficients b');
legend('b_1', 'b_2', 'b_3');

% True parameter values
true_parameters = [a; b];

% Display true and final parameter values
disp('True Parameters:');
disp(true_parameters);

disp('Final Parameter Estimates:');
disp(theta_ls);

% Covariance matrix for the final parameter vector estimate
cov_matrix = inv(H);

disp('Covariance Matrix:');
disp(cov_matrix);
