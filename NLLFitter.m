classdef NLLFitter < matlab.mixin.Copyable
    % NLLFitter is a simple utility class that computes the Negative
    % Log-Likelihood (NLL) and its derivatives for a given ModelBuilder instance.
    % It is designed to be used with MATLAB's optimization functions.

    properties
        inner_model (1,1) % The model to be fitted
        useGPU (1,1) logical = false
    end

    methods
        function self = NLLFitter(model_instance, pv)
            % The constructor takes an instance of a ModelBuilder class.
            arguments
                model_instance (1,1)
                pv.useGPU (1,1) logical = false
            end
            self.inner_model = model_instance;
            self.useGPU = pv.useGPU;
        end

        function p0 = estimate(self, x_data, y_data, varargin)
            % Estimates an initial guess (p0) for the combined parameter vector.
            % It first estimates the inner model's parameters, then uses the
            % resulting residuals to estimate sigma.
            
            if nargin > 1
                if isstring(x_data) || ischar(x_data)
                    % the input is not x or y but kwargs for
                    % inner_model.estimate
                    
                    varargin = [{x_data, y_data}, varargin];
                    x_data = self.inner_model.X;
                    y_data = self.inner_model.Y;                    
                
                end
            else

                x_data = self.inner_model.X;
                y_data = self.inner_model.Y;

            end
            
            % 1. Estimate the inner model's parameters
            p_inner_est = self.inner_model.estimate(x_data, y_data, varargin{:});
            
            % 2. Calculate the residuals of the inner model
            YHat = self.inner_model.predict(p_inner_est, X=x_data);
            R = y_data - YHat;
            
            % 3. Estimate sigma as the standard deviation of the residuals
            sigma_est = std(R, 0, 'all');
            
            % 4. Combine the parameters to form the initial guess vector
            p0 = [p_inner_est, sigma_est];
        end

        function [nll, G, H] = objective_function(self, p, W)

            arguments

                self
                p (1,:) = self.inner_model.P_
                W (:,:) = self.inner_model.W_ % residual weight

            end

            % This is the main objective function for an optimizer like fminunc.
            % It computes the NLL, its Gradient (G), and its Hessian (H) for
            % a given parameter vector 'p'.
                        
            if numel(self.inner_model.W) ~= numel(W) || any(self.inner_model.W ~= W,"all")
                % update W if provided
                self.inner_model.W = W;

            end
            % 1. Update the inner model with the current parameters
            p = self.set_parameters(p); % it will be converted to gpuArray if needed
            
            % 2. Compute the NLL value
            R = self.inner_model.wR(:);
            sigma = p(end);
            nll = sum( (R.^2) / (2 * sigma^2) + log(sigma) );
            nll = self.convertIfGPU_(nll, toGPU = false);
            % 3. Compute the Gradient vector
            if nargout > 1
                G = self.compute_gradient(p);
            end

            % 4. Compute the Hessian matrix
            if nargout > 2
                H = self.compute_hessian(p);
            end
        end

        function G = compute_gradient(self, p)
            % Computes the full gradient of the NLL function.
            
            
            % Update the inner model to ensure its derivatives are current
            p = self.set_parameters(p);            
            
            % Compute Jacobian if empty
            if isempty(self.inner_model.J_)
                self.inner_model.compute_jacobian();
            end

            J_inner = self.inner_model.J_;
            R_inner = self.inner_model.wR;
            n_sample = self.convertIfGPU_(numel(R_inner));
            R_inner = self.convertIfGPU_(sum(R_inner, 2));
            sigma   = p(end);

            if isempty(J_inner) || isempty(R_inner)
                G = []; return;
            end

            % Gradient with respect to inner model's parameters
            grad_vs_p_inner = - J_inner' * (R_inner ./ sigma^2);
            
            % Gradient with respect to sigma
            grad_vs_sigma = - sum(R_inner.^2) / sigma^3 + n_sample/sigma;
            
            G = [grad_vs_p_inner; grad_vs_sigma];
            G = self.convertIfGPU_(G, toGPU=false);
        end

        function H = compute_hessian(self, p)
            % Computes the full, exact Hessian of the NLL function.

            % Update the inner model to ensure its derivatives are current
            p = self.set_parameters(p);
            
            % must compute jacobian first
            if isempty(self.inner_model.J_) 
                self.compute_gradient(p);                
            end

            if isempty(self.inner_model.H_raw_)
                self.inner_model.compute_hessian();
            end
            
            
            % Jacobian
            J_inner = self.inner_model.J_;
            % Residuals
            R_inner = self.inner_model.wR;
            n_samp = numel(R_inner);
            R_sum = sum(R_inner,2);
            W = self.inner_model.W_; % weights
            W_sum_diag =  diag(sum(W, 2)); % H_pp_term1
            % Hessian tendor
            H_inner_raw = self.inner_model.H_raw_; % The per-sample Hessian tensor
            sigma = p(end);
            n_param = self.convertIfGPU_(self.inner_model.n_param);

            if isempty(J_inner) || isempty(R_inner) || isempty(H_inner_raw)
                H = []; return;
            end
            
            % --- Calculate blocks of the Hessian matrix ---
            
            % Top-left block: d²/dP² (includes the full second-order term)
            H_pp_term1 = (1/sigma^2) * (J_inner' * W_sum_diag * J_inner); % Gauss-Newton part
            
            % Flatten spatial dims of Hessian: (n_samp, n_param^2)
            if ndims(H_inner_raw) == 3
                [N, P, ~] = size(H_inner_raw);
                H_flat = reshape(H_inner_raw, N, P*P);
            else
                H_flat = H_inner_raw;

            end
           
            % Perform weighted sum via matrix multiplication:
            % (1 x n_samp) * (n_samp x P^2) -> (1 x P^2)
            dCost_dYhat = -R_sum / sigma^2;
            H_weighted_flat = dCost_dYhat' * H_flat;

            % Reshape back to (n_param, n_param)
            H_pp_term2 = reshape(H_weighted_flat, n_param, n_param);
            % 
            % H_pp_term2 = zeros(n_param);
            % 
            % for i = 1:n_samp
            %     H_pp_term2 = H_pp_term2 + dCost_dYhat(i) * squeeze(H_inner_raw(i, :, :));
            % end
            H_pp = H_pp_term1 + H_pp_term2;

            % Bottom-right block: d²/d(sigma)²
            H_ss = (3*sum(R_inner(:).^2))/sigma^4 - n_samp/sigma^2;
            
            % Off-diagonal blocks: d²/dPd(sigma)
            H_ps = J_inner' * ( (2*R_sum)/sigma^3 );
            
            % Assemble the full Hessian matrix
            H = [H_pp,  H_ps; ...
                 H_ps', H_ss];
            H = sparse(H);
            H = self.convertIfGPU_(H, toGPU = false);
        end

        % Estimate Bounds from data
        function [lb, ub] = estimate_parameter_bounds(self)

            mdl = self.inner_model;
            lb = [];
            ub = [];
            if ~isempty(mdl.lower_bounds)
                lb = [mdl.lower_bounds, 0]; % sigma cannot be smaller than 0
            end

            if ~isempty(mdl.upper_bounds)
                % if model fit is meaningful, residuals variance must not
                % exceed the sample variance
                ub = [mdl.upper_bounds, std(mdl.Y(:))];
            end

            [lb, ub] = self.convertIfGPU_(lb, ub, toGPU=false);

        end

        %% SET and GET methods

        function set.useGPU(self,val)
            
            arguments
                self
                val (1,1) logical
            end
            if val && ~canUseGPU
                warning("No available GPUs on the machine!!");
                return;
            end
            self.useGPU = val;
            self.set_gpu_();                  

        end

    end    

    methods (Access = protected)
        function cpObj = copyElement(self)
            % 1. Create a shallow copy of the parent object
            % This copies all value properties (numbers, structs) automatically.
            cpObj = copyElement@matlab.mixin.Copyable(self);
            
            % 2. Deep copy the specific handle property
            % We check isempty to avoid errors if ChildNode is not assigned.
            cpObj.inner_model = copy(self.inner_model);
            
        end
    end

    methods (Access = protected)

        function varargout = convertIfGPU_(self, varargin)

            varargout = cell(1,nargout);
            [varargout{:}] = self.inner_model.convertIfGPU_(varargin{:});

        end

        function set_gpu_(self)
                        
            if self.useGPU ~= self.inner_model.useGPU                
                self.inner_model.useGPU = self.useGPU;
            end  

        end
    end
    
    methods (Access = private)
        function p = set_parameters(self, p)
            % Helper function to distribute the composite parameter vector 'p'
            % to the inner model and this class's properties.
            n_inner_params = self.inner_model.n_param;
            if numel(p) ~= (n_inner_params + 1)
                error('Incorrect number of parameters. Expected %d.', n_inner_params + 1);
            end
            
            % This assignment triggers the on-demand computation engine
            % within the inner_model if the parameters have changed.
            self.inner_model.P = p(1:n_inner_params);
            p = [self.inner_model.P_, p(end)];

        end
    end
end