classdef ModelBuilder < matlab.mixin.Copyable
    % ModelBuilder is an abstract class that provides a framework for
    % building, fitting, and evaluating symbolic mathematical models.
    % It implements a "compute-on-demand" pattern, where numerical results
    % are calculated only when first requested. Changing core properties
    % automatically invalidates previous calculations.

    properties (Abstract)
        model sym % The main symbolic function for the model
        x sym     % The symbolic independent variable (e.g., 'x')
        y sym     % The symbolic dependent variable (e.g., 'y')
        y_hat sym % Symbolic representation of the model's prediction
    end

    properties (Abstract, Dependent)
        parameters (1,:) sym % A vector of the model's symbolic parameters
    end

    properties
        verbose (1,1) logical = true % Controls whether status messages are displayed
    end

    properties (Access = protected)
        % --- Hidden Storage Properties ---
        W_ (:, :) double = 1
        P_ (1, :) double = []
        X_ (:, 1) double = []
        Y_ (:, :) double = []

        jacobian_ sym = sym.empty()
        hessian_ sym = sym.empty()

        YHat_ (:, 1) double = []
        J_ (:,:) double = []
        H_raw_ (:,:,:) double = []
        G_ (:,1) double = []

    end

    properties (Dependent)
        % --- Public-Facing Properties with On-Demand Computation ---
        W, P, X, Y % weights, parameters, x_data, y_data
        jacobian, hessian
        YHat, J, H_raw % y_hat, jacobian matrix, hessian matrrix before summation
        G % gradient

        % --- Other Derived Properties ---
        n_param, n_sample, n_observation, sample_size
        H, R, wR, SSR % hessian matrix, residuals, weighted residuals, ssr

        null_model

    end

    %======================================================================
    % SET/GET METHODS: CORE DATA
    %======================================================================
    methods
        function set.W(self, value)
            if self.verbose; fprintf('>> W changed. Invalidating G, H_raw.\n'); end
            self.W_ = value;
            self.clear_computed_properties('W');
        end
        function val = get.W(self)
            val = self.W_;
        end

        function set.P(self, value)
            if self.verbose; fprintf('>> P changed. Invalidating YHat, J, G, H_raw.\n'); end
            self.P_ = value;
            self.clear_computed_properties('P');
        end
        function val = get.P(self)
            val = self.P_;
        end

        function set.X(self, value)
            if self.verbose; fprintf('>> X changed. Invalidating YHat, J, G, H_raw.\n'); end
            self.X_ = value;
            self.clear_computed_properties('X');
        end
        function val = get.X(self)
            val = self.X_;
        end

        function set.Y(self, value)
            if self.verbose; fprintf('>> Y changed. Invalidating G, H_raw.\n'); end
            self.Y_ = value;
            self.clear_computed_properties('Y');
        end

        function val = get.Y(self)
            val = self.Y_;
        end

        function mdl = get.null_model(self)

            % either the noise or the intercept (mean) model
            y = self.Y;
            if isempty(self.Y), mdl = struct(); return; end
            y = y.*self.W;

            avg = mean(self.Y);
            n_params0 = abs(avg) >= eps; % i.e., 0 if avg = 0, else 1
            aic0 = calculate_aic_(y, n_params0);
            if n_params0
                fit0 = avg;
                type0 = 'intercept';
            else
                fit0 = [];
                type0 = 'noise';
            end

            mdl = struct(type = type0, P = fit0, aic = aic0);

        end

    end

    %======================================================================
    % GET METHODS: SYMBOLIC & COMPUTED PROPERTIES
    %======================================================================
    methods
        % --- Symbolically Computed Properties ---
        function val = get.jacobian(self)
            if isempty(self.jacobian_)
                self.solve_jacobian();
            end
            val = self.jacobian_;
        end

        function val = get.hessian(self)
            if isempty(self.hessian_)
                self.solve_hessian();
            end
            val = self.hessian_;
        end

        % --- Numerically Computed Properties ---
        function val = get.YHat(self)
            if isempty(self.YHat_)
                self.YHat_ = self.predict();
            end
            val = self.YHat_;
        end

        function val = get.J(self)
            if isempty(self.J_)
                self.compute_jacobian();
            end
            val = self.J_;
        end

        function val = get.G(self)
            if isempty(self.G_)
                self.compute_gradient();
            end
            val = self.G_;
        end

        function val = get.H_raw(self)
            if isempty(self.H_raw_)
                self.compute_hessian();
            end
            val = self.H_raw_;
        end

    end

    %======================================================================
    % GET METHODS: OTHER DERIVED PROPERTIES
    %======================================================================
    methods
        function n = get.n_sample(self)
            n = size(self.X_, 1);  % check what to do when using weights
        end

        function n = get.sample_size(self)
            n = numel(self.Y_);
        end

        function n = get.n_param(self)
            n = numel(self.parameters);
        end

        function n = get.n_observation(self)
            n = size(self.Y_, 2);
        end

        function R = get.R(self)
            if isempty(self.YHat) || isempty(self.Y); R = []; return; end
            R =  self.Y - self.YHat;
        end

        function R = get.wR(self)
            if isempty(self.YHat) || isempty(self.Y); R = []; return; end
            R = self.W .* self.R;
        end
        function H = get.H(self)
            if isempty(self.H_raw); H = []; return; end
            H = squeeze(sum(self.H_raw, 1));
        end

        function SSR = get.SSR(self)
            if isempty(self.R); SSR = NaN; return; end
            SSR = sum(self.wR.^2);
        end
    end

    %======================================================================
    % CORE COMPUTATION METHODS
    %======================================================================
    methods
        function compute(self, varargin)
            % Computes specified numerical outputs, or all by default.
            % SYNTAX:
            %   compute(self)              % Computes all outputs
            %   compute(self, 'YHat', 'J') % Computes YHat and Jacobian

            if isempty(varargin)
                % Default to all if no specific properties are requested
                toCompute = ["YHat", "J", "G", "H_raw"];
            else
                toCompute = strings(size(varargin));
                for i = 1:numel(varargin)
                    toCompute(i) = validatestring(varargin{i}, ...
                        {'YHat', 'J', 'G', 'H_raw'}, 'compute', 'property to compute');
                end
            end

            if self.verbose; fprintf('--- Beginning On-Demand Computation ---\n'); end

            % Use unique to avoid computing the same property twice
            for prop = unique(toCompute, 'stable')
                if self.verbose; fprintf("Requesting '%s'...\n", prop); end
                % Accessing the property will trigger its on-demand get method
                self.(prop);
            end

            if self.verbose; fprintf('--- Computation Complete ---\n'); end
        end

        function YHat = predict(self, P, pv)

            arguments
                self
                P = self.P
                pv.X = self.X
            end

            try
            if self.verbose; fprintf('Computing YHat_...\n'); end
            tStart = tic;
            
            if isempty(P), P = self.P; end

            if isempty(pv.X), X = self.X; 
            else, X = pv.X; end

            if isempty(P) || isempty(X)
                self.YHat_ = []; YHat = [];
                if self.verbose; fprintf('\tCannot compute: P or X is empty.\n'); end
                return;
            end
            YHat = self.compute_(self.model, {self.parameters, self.x}, {P, X});
            if self.verbose; fprintf('\tDone. Elapsed time is %.4f seconds.\n', toc(tStart)); end
            catch e
                aa
            end
        end

        function y_sim = simulate(self, p, sigma, x_vals)
            % Simulates data from the model with added Gaussian noise.
            % This method does not alter the state of the object.
            arguments
                self
                p (1,:) double % The parameter values to use for simulation
                sigma (1,1) double % The standard deviation of the noise
                x_vals (:,1) double % The x-values to simulate at
            end

            % 1. Predict the clean signal using the model's formula
            y_clean = self.predict(p, x_vals);%self.compute_(self.model, {self.parameters, self.x}, {p, x_vals});

            % 2. Generate Gaussian noise with the specified sigma
            noise = randn(size(y_clean)) * sigma;

            % 3. Add the noise to the clean signal
            y_sim = y_clean + noise;
        end

        function solve_jacobian(self)
            if self.verbose;
                fprintf('Solving symbolic jacobian_...\n'); end
            tStart = tic;
            self.jacobian_ = self.solve_jacobian_(self.model, self.parameters);
            if self.verbose; fprintf('\tDone. Elapsed time is %.4f seconds.\n', toc(tStart)); end
        end

        function compute_jacobian(self)
            if self.verbose; fprintf('Computing J_...\n'); end
            tStart = tic;
            if isempty(self.X) || isempty(self.P)
                self.J_ = [];
                if self.verbose; fprintf('\tCannot compute: P or X is empty.\n'); end
                return;
            end
            self.J_ = self.compute_(self.jacobian, {self.parameters, self.x}, {self.P, self.X});
            if self.verbose; fprintf('\tDone. Elapsed time is %.4f seconds.\n', toc(tStart)); end
        end

        function compute_gradient(self)
            if self.verbose; fprintf('Computing G_...\n'); end
            tStart = tic;
            if isempty(self.R) || isempty(self.J)
                self.G_ = [];
                if self.verbose; fprintf('\tCannot compute: R or J is empty.\n'); end
                return;
            end
            self.G_ = 2 .* self.J' * (self.W .* self.R);
            if self.verbose; fprintf('\tDone. Elapsed time is %.4f seconds.\n', toc(tStart)); end
        end

        function solve_hessian(self)
            if self.verbose; fprintf('Solving symbolic hessian_...\n'); end
            tStart = tic;
            self.hessian_ = self.solve_jacobian_(self.jacobian, self.parameters);
            if self.verbose; fprintf('\tDone. Elapsed time is %.4f seconds.\n', toc(tStart)); end
        end

        function compute_hessian(self)
            if self.verbose; fprintf('Computing H_raw_...\n'); end
            tStart = tic;
            if isempty(self.R) || isempty(self.J)
                self.H_raw_ = [];
                if self.verbose; fprintf('\tCannot compute: R or J is empty.\n'); end
                return;
            end
            H_tensor = self.compute_(self.hessian, {self.parameters, self.x}, {self.P, self.X});
            if isscalar(unique(size(H_tensor)))

                self.H_raw_ = permute(repmat(H_tensor,[1,1,self.n_sample]),[3,1,2]);

            else
                self.H_raw_ = reshape(H_tensor, [self.n_sample, self.n_param, self.n_param]);
            end
            if self.verbose; fprintf('\tDone. Elapsed time is %.4f seconds.\n', toc(tStart)); end
        end


    end

    %======================================================================
    % UTILITY METHODS
    %======================================================================
    methods
        function mute(self, pv)
            % Sets verbose to false or toggles its state.
            arguments
                self
                pv.toggle (1,1) logical = false
            end
            if pv.toggle
                self.verbose = ~self.verbose;
            else
                self.verbose = false;
            end
            if self.verbose; fprintf('Verbose mode is ON.\n'); else; fprintf('Verbose mode is OFF.\n'); end
        end

        function unmute(self, pv)
            % Sets verbose to true or toggles its state.
            arguments
                self
                pv.toggle (1,1) logical = false
            end
            if pv.toggle
                self.verbose = ~self.verbose;
            else
                self.verbose = true;
            end
            if self.verbose; fprintf('Verbose mode is ON.\n'); else; fprintf('Verbose mode is OFF.\n'); end
        end
    end

    %======================================================================
    % PROTECTED HELPER METHODS
    %======================================================================
    methods (Access = protected)

        function clear_computed_properties(self, source_prop)
            % Invalidates downstream calculations when a core property changes.
            if any(strcmp(source_prop, {'Y', 'X'}))

                self.P_ = [];
                self.YHat_ = [];
                self.J_ = [];
                self.H_raw_ = [];
                self.G_ = [];

            end
            
            if any(strcmp(source_prop, {'P'}))
                
                self.YHat_ = [];
                self.J_ = [];
                self.H_raw_ = [];
                self.G_ = [];

            end

            if any(strcmp(source_prop, {'W'}))
                self.G_ = [];
                self.H_raw_ = [];
            end

            if any(strcmp(source_prop, {'Y', 'X'})) && isprop(self,{"lower_bounds_"})

                self.lower_bounds_ = [];
                self.upper_bounds_ = [];

            end
        end

    end


    methods (Access = protected, Static)

        function J = solve_jacobian_(varargin)
            % This is a protected wrapper for the Symbolic Math Toolbox's
            % jacobian function to prevent naming conflicts.
            J = jacobian(varargin{:}); % from Symbolic Math Toolbox
        end

        function y = compute_(expr, input_order, input_values)
            % Numerically evaluates any symbolic expression using matlabFunction.
            arguments
                expr {mustBeA(expr, 'sym')}
                input_order (1,:) cell
                input_values (1,:) cell
            end

            hasX = has(expr,'x');
            % add x to the terms we will subtract this later
            all_but_some_terms_hasX = sum(hasX,'all') && sum(hasX,'all') < numel(hasX);
            if all_but_some_terms_hasX
                expr = expr + sym('x');
            end
            func = matlabFunction(expr, 'Vars', input_order);

            func_str = func2str(func);
            % if what is a matrix make sure that it outputs 3D array when x
            % is a vector
            if ismatrix(expr) && any(hasX,'all') && contains(func_str,'reshape')

                % if contains reshape, last argument will be the shape
                % argument
                ii = find(func_str=='[',1,'last');

                func = str2func([func_str(1:ii), 'numel(x),', func_str(ii+1:end)]);

            end
            y = func(input_values{:});
            % if reshaped, the output wqill be 3D, make it 2D when x_value is
            % scalar
            y = squeeze(y);

            % x must always be input separately to be subtracted out for
            % the scalar terms
            if all_but_some_terms_hasX
                isX = cellfun(@(inp) all(has(inp,'x')), input_order);

                y = y - input_values{isX};
            end
        end

        % Misc
        function [y_interp, x_interp] = interpolate_breaks_(y, x)

            step = mode(diff(x));

            x_interp = (x(1):step:x(end))';
            y_interp = interp1(x, y, x_interp, 'linear');

        end

    end

end

%% LOCAL HELPER FUNCTIONS
function aic = calculate_aic_(residuals, n_param)

residuals = residuals(:);
n_sample = numel(residuals);
rss = sum(residuals.^2);
log_lik = -n_sample / 2 * (log(2*pi)+log(rss/n_sample) + 1);
aic = 2*n_param - 2*log_lik;
if n_sample / n_param < 40
    % correction for low sample sizes
    aic = aic + (2*n_param*(n_param + 1)) / (n_sample - n_param - 1);

end
end
