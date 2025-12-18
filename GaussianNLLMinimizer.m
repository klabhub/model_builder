classdef GaussianNLLMinimizer < NLLMinimizer
    % GaussianNLLMinimizer
    properties (Access = protected, Abstract)

        hyper_parameters_ sym

    end

    methods
        function self = GaussianNLLMinimizer(model_instance)
            
            self@NLLMinimizer(model_instance);
            self.hyper_parameters_ = sym('sigma');

        end
    end
    
    methods (Access = protected)
        function val = estimate_nuisance_parameter(self)
            % Estimate sigma from Weighted Residuals (wR)
            % wR = W .* R
            val = std(self.Model.wR, 0, 'all');
        end
        
        function [lb, ub] = get_nuisance_bounds(self)
            lb = 0; ub = std(self.Y(:)) * 10; % Relaxed upper bound
        end
        
        function nll = calculate_nll(self, p)
            % We pass raw R and W to handle the math explicitly
            R = self.inner_model.R; 
            W = self.inner_model.W;
            nll = gausserr.calculate_nll(p, R, W);
        end
        
        function G = calculate_gradient(self, p)
            J = self.inner_model.J;
            R = self.inner_model.R; 
            W = self.inner_model.W;
            
            if isempty(J), G = []; return; end
            
            G = gausserr.calculate_gradient(p, J, R, W);
        end
        
        function H = calculate_hessian(self, p)
            J = self.inner_model.J;
            R = self.inner_model.R;
            W = self.inner_model.W;
            H_raw = self.inner_model.H_raw;
            
            if isempty(J), H = []; return; end
            
            H = gausserr.calculate_hessian(p, J, R, W, H_raw);
        end
    end
end