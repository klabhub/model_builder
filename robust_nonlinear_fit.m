function [fitted_params, ii, isConverged] = robust_nonlinear_fit(...
    fitter, p0, pv)

arguments

    fitter
    p0 (1,:) {mustBeNumeric}
    pv.max_iter (1,1) {mustBePositive, mustBeInteger} = 10
    
    pv.abs_z_threshold = 3.33
    pv.lb {mustBeNumeric} = []
    pv.ub {mustBeNumeric} = []
    pv.A {mustBeNumeric} = []
    pv.b {mustBeNumeric} = []
    pv.Aeq {mustBeNumeric} = []
    pv.beq {mustBeNumeric} = []
    pv.nonlcon {mustBeNumeric} = []
    pv.options = optimoptions('fmincon') % optimizer options
    pv.conv_tol = 1e-6 % param convergence tolerance

    pv.plot = false

end

% res_scalar = 1.4826;
current_params = p0;

n = fitter.inner_model.n_sample;

isExcld = false([n,1]);

if pv.plot
    figure;
    plot(fitter.inner_model.X, fitter.inner_model.Y, Linewidth=2);
    hold on
end
for ii = 1:pv.max_iter % start iterations

    weights = ones([n,1]);
    % weights(isExcld) = 0;

    last_params = current_params;
    % Nonlinear fit with the residual function with weights
    current_params = fmincon(@(p) fitter.objective_function(p, weights), ...
        current_params, pv.A, pv.b, pv.Aeq, pv.beq, pv.lb, pv.ub, ...
        pv.nonlcon, pv.options);
    % actual residuals without weights
    R = detrend(fitter.inner_model.R);
    R_z = do.robust_z(R);    
    isExcld = isExcld | R_z >= pv.abs_z_threshold;

    if pv.plot
       
        plot(fitter.inner_model.X, fitter.inner_model.predict());
        if sum(isExcld)
            xline(fitter.inner_model.X(isExcld),Alpha=.2);
        end
        drawnow()
    end
    
    % check for convergence between the first and last step
    isConverged = sum((current_params - last_params).^2) / (sum(last_params.^2) + eps) < pv.conv_tol;
    if isConverged; break; end
    
end
fitted_params = current_params;

end