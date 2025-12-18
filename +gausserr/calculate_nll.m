function nll = calculate_nll(p, R, W)

arguments (Input)
    p (1,:) double {mustBeReal}
    R (:,:) double {mustBeReal}
    W (:,:) double {mustBeReal} = ones(size(R))
end

arguments (Output)
    nll (1,1) double
end

%{
CALCULATE_NLL Computes the Negative Log-Likelihood for Gaussian Errors.

Objective:
    Minimizing the NLL for a Gaussian distribution with parameter sigma.
    The residuals are assumed to be weighted by W.
    
Derivation:
    Likelihood L = Product( (1/sqrt(2*pi*sigma^2)) * exp( - (W*R)^2 / (2*sigma^2) ) )
    Log-Likelihood LL = Sum( -ln(sigma) - (W*R)^2 / (2*sigma^2) ) + Constant
    Negative LL (NLL) = Sum( ln(sigma) + (W*R)^2 / (2*sigma^2) )

Formula:
    NLL = Sum[ (W_i * R_i)^2 / (2 * sigma^2) + ln(sigma) ]

Inputs:
    p - Parameter vector [parameters, sigma]
    R - Residual vector (Y - YHat)
    W - (Optional) Weight vector/matrix.
%}

sigma = p(end);

% Apply residual weights
wR = R(:) .* W(:);

% Compute NLL
% Note: We sum the log(sigma) term for each data point to keep scaling consistent
nll = sum( (wR.^2) / (2 * sigma^2) + log(sigma) );
end