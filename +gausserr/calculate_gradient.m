function G = calculate_gradient(p, J, R, W)
arguments (Input)
    p (1,:) double {mustBeReal}
    J (:,:) double {mustBeReal}
    R (:,:) double {mustBeReal}
    W (:,:) double {mustBeReal} = ones(size(R))
end
arguments (Output)
    G (:,1) double
end

%{
CALCULATE_GRADIENT Computes the gradient of the Gaussian NLL.

Derivations:
    Let E = (W*R)^2 / (2*sigma^2) + ln(sigma)
    Total NLL = Sum(E)

1. Gradient w.r.t Model Parameters (theta):
    d(NLL)/d(theta) = d(NLL)/dR * dR/d(theta)
    Since R = Y - YHat(theta), dR/d(theta) = -J
    d(NLL)/dR = (W^2 * R) / sigma^2
    Result: -J' * (W^2 * R) / sigma^2

2. Gradient w.r.t Sigma:
    d(NLL)/d(sigma) = Sum( d/d(sigma) [ (W*R)^2 * 0.5 * sigma^-2 + ln(sigma) ] )
    = Sum( (W*R)^2 * (-sigma^-3) + 1/sigma )
    = -Sum((W*R)^2)/sigma^3 + N/sigma
%}

sigma = p(end);

R = R(:);

W2 = W(:).^2; % Weights applied to the SQUARE of residuals

n_sample = numel(R);

% Effective residual for gradient calculation: W^2 * R
R_eff = W2 .* R;

% dNLL/dP
grad_vs_p_inner = -J' * (R_eff ./ sigma^2);

% dNLL/dSigma
% Note: sum(R_eff .* R) is equivalent to sum(W^2 * R^2)
grad_vs_sigma = -sum(R_eff .* R) / sigma^3 + n_sample/sigma;

G = [grad_vs_p_inner; grad_vs_sigma];
end