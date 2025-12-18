function H = calculate_hessian(p, J, H_tensor, R, W)
arguments (Input)
    p (1,:) double {mustBeReal}
    J (:,:) double {mustBeReal}
    H_tensor (:,:,:) double {mustBeReal}
    R (:,:) double {mustBeReal}
    W (:,:) double {mustBeReal} = ones(size(R))
end
arguments (Output)
    H (:,:) double
end

%{
CALCULATE_HESSIAN Computes the Hessian matrix of the Gaussian NLL.

Blocks:
1. H_pp (Params vs Params):
   d/d(theta) [ -J' * (W^2*R)/sigma^2 ]
   = (1/sigma^2) * J' * diag(W^2) * J  (Gauss-Newton Term, from dR/dtheta)
     + Term involving dJ/dtheta (Newton correction using H_raw)

2. H_ss (Sigma vs Sigma):
   d/d(sigma) [ -Sum(W^2 R^2)/sigma^3 + N/sigma ]
   = 3 * Sum(W^2 R^2)/sigma^4 - N/sigma^2

3. H_ps (Params vs Sigma):
   d/d(sigma) [ -J' * (W^2 R)/sigma^2 ]
   = -J' * (W^2 R) * (-2/sigma^3)
   = J' * (2 * W^2 * R) / sigma^3
%}

sigma = p(end);
R = R(:);
n_sample = numel(R);
n_param = size(J, 2);
W2 = W(:).^2;

% --- H_pp (Parameter-Parameter Block) ---
% 1. Gauss-Newton approximation term
W2_diag = diag(W2);
H_pp_term1 = (1/sigma^2) * (J' * W2_diag * J);

% 2. Full Newton correction term (Curvature of the model)
% d(Gradient)/d(theta) involves second derivative of residuals
H_pp_term2 = zeros(n_param);
dCost_dYhat = -(W2 .* R) / sigma^2;

for i = 1:n_sample
    H_pp_term2 = H_pp_term2 + dCost_dYhat(i) * squeeze(H_tensor(i, :, :));
end
H_pp = H_pp_term1 + H_pp_term2;

% --- H_ss (Sigma-Sigma Block) ---
SSR = sum(W2 .* R.^2);
H_ss = (3 * SSR) / sigma^4 - n_sample / sigma^2;

% --- H_ps (Parameter-Sigma Block) ---
H_ps = J' * ( (2 * (W2 .* R)) / sigma^3 );

% Assemble Matrix
H = [H_pp,  H_ps; ...
    H_ps', H_ss];
end