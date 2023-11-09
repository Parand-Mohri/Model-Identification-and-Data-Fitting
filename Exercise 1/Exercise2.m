% Load the new data
load('dataMIDFLR1_2.mat');

% Perform least squares regression
X = [ones(size(x)), x, x.^2];
b_hat = X \ y;
y_hat = X * b_hat;

plot(x, y, 'o', x, y_hat, '-')
legend('Original data', 'Predicted values');

% Calculate the covariance matrix of the estimator beta_LS
N = length(y);
n = size(X, 2);
s2_LS = (1 / (N - n)) * sum((y - y_hat).^2);
PhiTPhi_inv = inv(X' * X);
cov_beta_LS = s2_LS * PhiTPhi_inv;

% Confidence intervals and hypothesis testing
alpha = 0.05;
c_q = 1.96;
CI_beta = [b_hat - c_q * sqrt(diag(cov_beta_LS)), b_hat + c_q * sqrt(diag(cov_beta_LS))];
reject_H0 = abs(b_hat) > c_q * sqrt(diag(cov_beta_LS));

% Display results
disp('95% Confidence Intervals for each component of beta:');
disp(CI_beta);

disp('Hypothesis Testing (H0: beta_j = 0 at 5% level):');
disp('Reject H0?');
disp(reject_H0);
%% 
% Assuming you have yt, Phi, and cov_beta_LS from the previous code

% Initialize arrays to store upper and lower bounds of confidence intervals
lower_bound = zeros(size(y));
upper_bound = zeros(size(y));

% Calculate confidence intervals for each data point
for i = 1:length(y)
    Var_yt = X(i, :) * cov_beta_LS * X(i, :)';
    confidence_interval = c_q * sqrt(Var_yt);
    
    lower_bound(i) = y_hat(i) - confidence_interval;
    upper_bound(i) = y_hat(i) + confidence_interval;
end

% Plot the data points and confidence intervals
figure;
scatter(x, y, 'o');
hold on;
plot(x, y_hat, '-', 'LineWidth', 2, 'DisplayName', 'Estimated values');
plot(x, lower_bound, '--', 'LineWidth', 1.5, 'DisplayName', 'Lower Bound (95% CI)');
plot(x, upper_bound, '--', 'LineWidth', 1.5, 'DisplayName', 'Upper Bound (95% CI)');
legend();
xlabel('x');
ylabel('y');
title('Data Points and Confidence Intervals');
hold off;


