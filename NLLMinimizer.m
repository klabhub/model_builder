classdef (Abstract) NLLMinimizer < matlab.mixin.Copyable
    % NLLMinimizer defines the workflow for NLL optimization.
    % It ensures the inner_model is synchronized with the optimizer's parameters
    % and weights, but delegates specific math to concrete subclasses.
    
    properties
        Model (1,1) ModelBuilder % The model to be fitted (ModelBuilder)
    end

    properties (Dependent)

        P % all params [modelP, hyperP]
        hyperP % free params from the minimizer

        n_hyper_param
        n_param

        hyper_parameters
        parameters
    end

    properties (Access = protected)

        hyperP_        

    end

    properties (Access = protected, Abstract)

        hyper_parameters_ sym      

    end
    
    methods
        function self = NLLMinimizer(model_instance)
            arguments
                model_instance (1,1)
            end
            self.Model = model_instance;
        end
        
        function p0 = estimate(self, x_data, y_data, varargin)
            % Template method for estimating initial parameters.
            
            if nargin > 1
                if isstring(x_data) || ischar(x_data)
                    varargin = [{x_data, y_data}, varargin];
                    x_data = self.Model.X;
                    y_data = self.Model.Y;                    
                end
            else
                x_data = self.Model.X;
                y_data = self.Model.Y;
            end
            
            % 1. Estimate Model Parameters
            p_inner_est = self.Model.estimate(x_data, y_data, varargin{:});
            
            % 2. Calculate Residuals
            YHat = self.Model.predict(p_inner_est, x_data);
            R = y_data - YHat;
            W = self.Model.W;
            
            % 3. Estimate Nuisance Parameter (Delegated)
            nuisance_est = self.estimate_nuisance_parameter(R, YHat, y_data, W);
            
            % 4. Combine
            p0 = [p_inner_est, nuisance_est];
        end
        
        function [nll, G, H] = objective_function(self, p, W)
            % The main entry point for the optimizer.
            arguments
                self
                p (1,:)
                W (:,:) = self.Model.W
            end
            
            % 1. Update Weights if provided and different
            if numel(self.Model.W) ~= numel(W) || any(self.Model.W ~= W, "all")
                self.Model.W = W;
            end
            
            % 2. Push parameters to inner model (updates J, YHat, etc.)
            self.set_parameters(p);
            
            % 3. Calculate NLL
            nll = self.calculate_nll(p);
            
            % 4. Calculate Gradient
            if nargout > 1
                G = self.calculate_gradient(p);
            end
            
            % 5. Calculate Hessian
            if nargout > 2
                H = self.calculate_hessian(p);
            end
            
        end
        
        function [lb, ub] = estimate_parameter_bounds(self)
            mdl = self.Model;
            lb_inner = []; ub_inner = [];
            
            if ~isempty(mdl.lower_bounds), lb_inner = mdl.lower_bounds; end
            if ~isempty(mdl.upper_bounds), ub_inner = mdl.upper_bounds; end
            
            [lb_nuisance, ub_nuisance] = self.get_nuisance_bounds(mdl.Y);
            
            lb = [lb_inner, lb_nuisance];
            ub = [ub_inner, ub_nuisance];
        end
    end

    % --------------------------
    % -----Get/Set Methods------
    % --------------------------

    methods

        function p = get.hyper_parameters(self)

            p = self.hyper_parameters_;

        end

        function p = get.hyperP(self)

            p = self.hyperP_;

        end

        function set.hyperP(self, val)
            
            assert(all(size(val) == size(self.hyper_parameters)),...
                "Hyperparameters were attempted to an array of incompatible size.");

            self.hyperP_ = val;

        end
        
        function p = get.P(self)

            p = [self.Model.P, self.hyperP];

        end

        function p = get.n_hyper_param(self)

            p = numel(self.hyper_parameters);

        end

        function p = get.n_param(self)

            p = self.Model.n_param + self.n_hyper_param;

        end

        function p = get.parameters(self)

            p = [self.Model.parameters, self.hyper_parameters];
        end

    end
    
    methods (Access = protected)
        function set_parameters(self, p)

            arguments
                self
                p (1,:)
            end
            % Helper function to distribute the composite parameter vector 'p'
            % to the inner model and this class's properties.
            
            if numel(p) ~= self.n_param
                error('Incorrect number of parameters. Expected %d.', n_inner_params + 1);
            end

            n_inner_params = self.Model.n_param;
            % This assignment triggers the on-demand computation engine
            % within the inner_model if the parameters have changed.
            self.Model.P = p(1:n_inner_params);
            self.hyperP = p((n_inner_params+1):end);
        end
    end
    
    % --- Abstract Methods ---
    methods (Abstract, Access = protected)
        val = estimate_nuisance_parameter(self, R, YHat, Y, W)
        [lb, ub] = get_nuisance_bounds(self, Y)
        
        val = calculate_nll(self, p)
        val = calculate_gradient(self, p)
        val = calculate_hessian(self, p)
    end
end