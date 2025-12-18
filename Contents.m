% MODEL_BUILDER creates analytical models and fits data using MLE
% Version 1.0.0 (Fall 2025)
% This library creates models using Symbolic Math Library to fit to a
% dataset using Maximum Likelihood Estimator. If the model is
% differentiable it automatically derives partial derivatives and
% calculates Jacobian and Hessian matrices.
%
% Core Classes
%   ModelBuilder    - Builds models using Symbolic Math toolbox
%   NLLFitter       - Negative LogLikelihood Optimizer
%
% Subclasses with a concrete model
%   SumOfGaussians  - Models the sum of multiple Gaussians each having
%                     different centers, sigma, etc. for peak detection
%   ExponentialPowerLaw - Models the aperiodic component of power spectra
%
% Implementation functions
%   robust_nonlinear_fit    - Iteratively fits a model to data until
%                             convergence achieved.
%
% Dependencies - Mathworks Toolboxes
%   The following Add-Ons are required
%       * Symbolic Math Toolbox
%
% Dependencies - Third-Party Submodules
%   Ensure to install initiate submodules recursively after cloning the
%   repository:
%       * <a href="https://github.com/mert-ozkan/dataop.git"><dataop></a> - (Simple data operations toolbox)
%
% Copyright 2025 Mert Ozkan (KLab) @CMBN, Rutgers University
% License: MIT