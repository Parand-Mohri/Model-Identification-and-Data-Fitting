%% Exercise1 (a)Find the least square estimator
load('dataMIDFLR1.mat');
X = [ones(size(x)), x, x.^2];

% b_hat = regress(y, X);
b_hat = X \ y;
y_hat = X * b_hat;

plot(x, y, 'o', x, y_hat, '-')
legend('Original data', 'Predicted values');
%% (b) Find the covariance 
N = length(y); % Number of data points
n = size(X, 2); % Number of parameters

% Calculate s^2_LS
s2_LS = (2 / (N - n)) * var(y - y_hat);

% Calculate (Phi^T Phi)^-1
PhiTPhi_inv = inv(X' * X);

% Calculate cov(\beta_LS)
cov_beta_LS = s2_LS * PhiTPhi_inv;

% Display the covariance matrix
disp('Covariance matrix of the estimator beta_LS:')
disp(cov_beta_LS);
%% (c) Exploit cov(βˆLS) to obtain a 95% confidence interval
% The null hypothesis can be rejected if 0 is NOT in the interval 
alpha = 0.05; % Significance level
c_q = 1.96; % Critical value for a 95% confidence interval

% Confidence intervals for each component of beta
CI_beta = [b_hat - c_q * sqrt(diag(cov_beta_LS)), b_hat + c_q * sqrt(diag(cov_beta_LS))];

% Hypothesis testing
reject_H0 = abs(b_hat) > c_q * sqrt(diag(cov_beta_LS));

% Display results
disp('95% Confidence Intervals for each component of beta:')
disp(CI_beta);

disp('Hypothesis Testing (H0: beta_j = 0 at 5% level):')
disp('Reject H0?');
disp(reject_H0);
%% (d) Build a new model by retaining only the significant βˆj
% Assuming you have the reject_H0 vector from the previous code
significant_indices = find(reject_H0);

% Retain only the significant components in the design matrix
X_new = X(:, significant_indices);

% Estimate the new parameters
b_hat_new = X_new \ y;

% Display the old and new parameter estimates
disp('Old Parameter Estimates (b_hat):');
disp(b_hat);

disp('New Parameter Estimates (b_hat_new):');
disp(b_hat_new);




