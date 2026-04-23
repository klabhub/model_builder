classdef ModelBuilder < matlab.mixin.Copyable
    % ModelBuilder is an abstract class that provides a framework for
    % building, fitting, and evaluating symbolic mathematical models.
    % It implements a "compute-on-demand" pattern, where numerical results
    % are calculated only when first requested. Changing core properties
    % automatically invalidates previous calculations.

    properties (Abstract)
        model sym % The main symbolic function for the model
    end

    properties (Abstract, Dependent)
        parameters (1,:) sym % A vector of the model's symbolic parameters
    end

    properties
        verbose (1,1) logical = true % Controls whether status messages are displayed
        useGPU (1,1) logical = false
        x sym = sym('x')    % The symbolic independent variable (e.g., 'x')
        y sym = sym('y')    % The symbolic dependent variable (e.g., 'y')
        y_hat = sym('y_hat') % Symbolic representation of the model's prediction
    end

    properties (SetAccess = protected)
        % --- Hidden Storage Properties ---
        W_ (:, :) = 1
        P_ (1, :) = []
        X_ (:, 1) = []
        Y_ (:, :) = []

        jacobian_ sym = sym.empty()
        hessian_ sym = sym.empty()

        YHat_ (:, 1) = []
        J_ (:,:) = []
        H_raw_ (:,:,:) = []
        G_ (:,1) = []

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
        R_
        null_model
        NULL_
    end

    % Cached Functions from Symbolic Expressions
    properties (Access = protected)

        model_func_ function_handle
        jacobian_func_ function_handle
        hessian_func_ function_handle
        null_vec_ = sym('null') % to preserve the shape of hessian and jacobian tensors


    end

    %======================================================================
    % SET/GET METHODS: CORE DATA
    %======================================================================
    methods
        function set.W(self, value)
            self.set_data_prop_('W', value);
        end
        function val = get.W(self)
            val = self.convertIfGPU_(self.W_, toGPU=false);
        end

        function set.P(self, value)
            self.set_data_prop_('P', value);
        end
        function val = get.P(self)
            val = self.convertIfGPU_(self.P_, toGPU=false);
        end

        function set.X(self, value)
            self.set_data_prop_('X', value);
        end
        function val = get.X(self)
            val = self.convertIfGPU_(self.X_, toGPU=false);
        end

        function set.Y(self, value)
            self.set_data_prop_('Y', value);
        end

        function val = get.Y(self)
            val = self.convertIfGPU_(self.Y_, toGPU=false);
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
            hasNull = has(val, self.null_vec_);
            if any(hasNull)
                val(hasNull) = val(hasNull) - self.null_vec_;
            end
        end

        function val = get.hessian(self)
            if isempty(self.hessian_)
                self.solve_hessian();
            end
            val = self.hessian_;
            hasNull = has(val, self.null_vec_);
            if any(hasNull)
                val(hasNull) = val(hasNull) - self.null_vec_;
            end
        end

        % --- Numerically Computed Properties ---
        function val = get.YHat(self)
            if isempty(self.YHat_)
                self.YHat_ = self.predict();
            end
            % convert to cpu if useGPU
            val = self.convertIfGPU_(self.YHat_, toGPU=false);
        end

        function val = get.J(self)
            if isempty(self.J_)
                self.compute_jacobian();
            end
            val = self.convertIfGPU_(self.J_, toGPU=false);
        end

        function val = get.G(self)
            if isempty(self.G_)
                self.compute_gradient();
            end
            val = self.convertIfGPU_(self.G_, toGPU=false);
        end

        function val = get.H_raw(self)
            if isempty(self.H_raw_)
                self.compute_hessian();
            end
            val = self.convertIfGPU_(self.H_raw_, toGPU=false);
        end

        function mdl = get.null_model(self)

            % either the noise or the intercept (mean) model
            y_data = self.Y;
            if isempty(self.Y), mdl = struct(); return; end
            if ~isempty(self.W_)
                y_data = y_data.*self.W_;
            end

            avg = mean(y_data(:));
            n_params0 = abs(avg) >= eps; % i.e., 0 if avg = 0, else 1
            aic0 = calculate_aic_(y_data(:), n_params0);
            if n_params0
                fit0 = avg;
                type0 = 'intercept';
            else
                fit0 = [];
                type0 = 'noise';
            end

            mdl = struct(type = type0, ...
                P = self.convertIfGPU_(fit0, toGPU = false), ...
                aic = self.convertIfGPU_(aic0,toGPU=false));

        end

        function set.useGPU(self, val)

            arguments
                self
                val (1,1) logical
            end

            if self.useGPU ~= val
                self.useGPU = val;
                % must re-store values to convert to or from gpuArrays
                args = cellfun(@(field) {field, self.(field)}, ...
                    {'Y', 'X', 'P', 'W'}, 'UniformOutput', false);
                args = [args{:}];
                self.set_data_prop_(args{:}, force=true);
                % force = true to run even if useGPU=false

            end

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
            R = self.convertIfGPU_(self.R_, toGPU=false);
        end

        function R = get.R_(self)

            if isempty(self.YHat) || isempty(self.Y)
                R = []; return;
            end
            R =  self.Y_ - self.YHat_;

        end

        function R = get.wR(self)

            if isempty(self.YHat) || isempty(self.Y); R = []; return; end
            if isempty(self.W_), self.W = 1; end
            R = self.W_ .* self.R_;

        end
        function H = get.H(self)
            if isempty(self.H_raw); H = []; return; end
            H = squeeze(sum(self.H_raw, 1));
        end

        function SSR = get.SSR(self)
            if isempty(self.R); SSR = NaN; return; end
            SSR = sum(self.wR.^2);
        end

        function n = get.NULL_(self)
            n = self.convertIfGPU_(zeros(size(self.X_)));
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
                P = self.P_
                pv.X = self.X_
            end

            % data
            if self.verbose; fprintf('Computing YHat_...\n'); end
            tStart = tic;

            P = self.convertIfGPU_(P);
            x_data = self.convertIfGPU_(pv.X);

            if isempty(P) || isempty(x_data)
                self.YHat_ = []; YHat = [];
                if self.verbose; fprintf('\tCannot compute: P or X is empty.\n'); end
                return;
            end
            YHat = self.model_func_(P, x_data);
            % YHat = self.compute_(self.model, {self.parameters, self.x}, {P, x_data});
            YHat = self.convertIfGPU_(YHat, toGPU=false);
            if self.verbose; fprintf('\tDone. Elapsed time is %.4f seconds.\n', toc(tStart)); end
        end

        % function y_sim = simulate(self, p, sigma, x_vals)
        %     % Simulates data from the model with added Gaussian noise.
        %     % This method does not alter the state of the object.
        %     arguments
        %         self
        %         p (1,:) double % The parameter values to use for simulation
        %         sigma (1,1) double % The standard deviation of the noise
        %         x_vals (:,1) double % The x-values to simulate at
        %     end
        %
        %     % 1. Predict the clean signal using the model's formula
        %     y_clean = self.predict(p, X = x_vals);%self.compute_(self.model, {self.parameters, self.x}, {p, x_vals});
        %
        %     % 2. Generate Gaussian noise with the specified sigma
        %     noise = randn(size(y_clean)) * sigma;
        %
        %     % 3. Add the noise to the clean signal
        %     y_sim = y_clean + noise;
        % end
        function [expr, input_order] = stabilize_expression_(self, expr, addNull)

            arguments
                self
                expr
                addNull = false
            end
            % if expr is a matrix and it contains x in some elements
            % but not all null_vec_ must be added to the terms to ensure
            % shape of the ouput later is consistent
            input_order = {self.parameters, self.x};
            hasX = has(expr, self.x);
            if addNull && ~all(hasX(:))

                expr(~hasX) = expr(~hasX) + self.null_vec_;
                input_order = [input_order,{self.null_vec_}];

            end

        end

        function solve_model(self)

            self.model_func_ = self.expr2func_(self.model, ...
                {self.parameters, self.x});

        end

        function solve_jacobian(self)
            if self.verbose
                fprintf('Solving symbolic jacobian_...\n');
            end
            tStart = tic;
            j = self.solve_jacobian_(self.model, self.parameters);
            [self.jacobian_, input_order] = self.stabilize_expression_(j, ...
                true);% last argument is addNull
            % Create matlab function
            self.jacobian_func_ = self.expr2func_(self.jacobian_, ...
                input_order);
            if self.verbose; fprintf('\tDone. Elapsed time is %.4f seconds.\n', toc(tStart)); end
        end

        function compute_jacobian(self)

            if self.verbose; fprintf('Computing J_...\n'); end
            tStart = tic;
            if isempty(self.X_) || isempty(self.P_)
                self.J_ = self.convertIfGPU_([]);
                if self.verbose; fprintf('\tCannot compute: P or X is empty.\n'); end
                return;
            end
            inputs = {self.P_, self.X_};
            if nargin(self.jacobian_func_) == 3
                inputs = [inputs, self.NULL_];
            end
            self.J_ = self.jacobian_func_(inputs{:});
            % self.J_ = self.compute_(self.jacobian, {self.parameters, self.x}, {self.P_, self.X_});
            if self.verbose; fprintf('\tDone. Elapsed time is %.4f seconds.\n', toc(tStart)); end
        end

        function compute_gradient(self)
            if self.verbose; fprintf('Computing G_...\n'); end
            tStart = tic;
            if isempty(self.J_) % compute jacobian first

                self.compute_jacobian();

            end
            % If no residuals, cannot compute gradient
            if isempty(self.R_)
                self.G_ = self.convertIfGPU_([]);
                if self.verbose
                    fprintf('\tCannot compute: R or J is empty.\n');
                end
                return;
            end
            self.G_ = 2 .* self.J_' * (self.W_ .* self.R_);
            if self.verbose; fprintf('\tDone. Elapsed time is %.4f seconds.\n', toc(tStart)); end
        end

        function solve_hessian(self)
            if self.verbose; fprintf('Solving symbolic hessian_...\n'); end
            tStart = tic;
            h = self.solve_jacobian_(self.jacobian, self.parameters);
            [self.hessian_, input_order] = self.stabilize_expression_(h, true);% last argument is addNull
            self.hessian_func_ = self.expr2func_(self.hessian_, input_order);
            if self.verbose; fprintf('\tDone. Elapsed time is %.4f seconds.\n', toc(tStart)); end
        end

        function compute_hessian(self)
            if self.verbose; fprintf('Computing H_raw_...\n'); end
            n_samp = self.convertIfGPU_(self.n_sample);
            n_params = self.convertIfGPU_(self.n_param);
            tStart = tic;
            if isempty(self.J_) % compute Jacobian first

                self.compute_jacobian();

            end
            if isempty(self.R_)
                self.H_raw_ = self.convertIfGPU_([]);
                if self.verbose; fprintf('\tCannot compute: R is empty.\n'); end
                return;
            end

            inputs = {self.P_, self.X_};
            if nargin(self.hessian_func_) == 3
                inputs = [inputs, self.NULL_];
            end
            H_tensor = self.hessian_func_(inputs{:});
            % H_tensor = self.compute_(self.hessian, {self.parameters, self.x}, {self.P_, self.X_});
            self.H_raw_ = H_tensor;
            % if isscalar(unique(size(H_tensor)))
            % 
            %     reps = [1,1,n_samp];
            %     dims = self.convertIfGPU_([3,1,2]);
            %     self.H_raw_ = permute(repmat(H_tensor,reps),dims);
            % 
            % else
            %     self.H_raw_ = reshape(H_tensor, [n_samp, n_params, n_params]);
            % end
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

        function arr_out = convertIfGPU_(self, arr, pv)
            arguments (Input)
                self
            end

            arguments (Input, Repeating)
                arr
            end

            arguments (Input)
                pv.toGPU = true
                pv.force = false % force convert
                % used if useGPU is set after the class instance was
                % initiated
            end

            arguments (Output, Repeating)
                arr_out
            end

            if ~self.useGPU && ~pv.force, arr_out = arr; return; end

            if pv.toGPU
                func = @(x) gpuArray(x);
            else
                func = @(x) gather(x);
            end

            arr_out = cellfun(@(x) func(x), arr, UniformOutput=false);


        end
    end

    %======================================================================
    % PROTECTED HELPER METHODS
    %======================================================================
    methods (Access = protected)

        function clear_computed_properties(self, source_prop)
            % Invalidates downstream calculations when a core property changes.
            switch source_prop
                case {'Y', 'X'}

                    cleared_props = {'P', 'YHat', 'J', 'H_raw', 'G', 'W'};
                    if isprop(self,{"lower_bounds_"})

                        cleared_props = [cleared_props, {'lower_bounds', 'upper_bounds'}];

                    end

                case 'P'

                    cleared_props = {'YHat', 'J', 'H_raw', 'G'};

                case 'W'
                    cleared_props = {'H_raw', 'G'};

                otherwise

                    error("The property `%s` is not associated with a computed property.", source_prop);

            end

            for ii = 1:numel(cleared_props)

                self.([cleared_props{ii},'_']) = [];

            end

            if self.verbose
                fprintf("Reset %s.\n", join(string(cleared_props),', '));
            end

            %
            % if any(strcmp(source_prop, {'Y', 'X'}))
            %
            %     self.P_ = [];
            %     self.YHat_ = [];
            %     self.J_ = [];
            %     self.H_raw_ = [];
            %     self.G_ = [];
            %     self.W_ = [];
            %
            %
            % end
            %
            % if any(strcmp(source_prop, {'P'}))
            %
            %     self.YHat_ = [];
            %     self.J_ = [];
            %     self.H_raw_ = [];
            %     self.G_ = [];
            %
            % end
            %
            % if any(strcmp(source_prop, {'W'}))
            %     self.G_ = [];
            %     self.H_raw_ = [];
            % end
            %
            % if any(strcmp(source_prop, {'Y', 'X'})) && isprop(self,{"lower_bounds_"})
            %
            %     self.lower_bounds_ = [];
            %     self.upper_bounds_ = [];
            %
            % end
            %
            % if self.verbose;
            %     fprintf("Resetting %s.\n", join(string(cleared_props),', '));
            % end
        end

        function set_data_prop_(self, prop_name, value, pv)


            arguments
                self
            end

            arguments (Repeating)
                prop_name %char
                value %{mustBeNumeric}
            end

            arguments
                pv.toGPU = self.useGPU
                pv.force = false; % in case useGPU toggled
            end
            try
                n_props = numel(prop_name);
                for ii = 1:n_props
                    stored_name = sprintf("%s_",prop_name{ii});
                    self.(stored_name) = self.convertIfGPU_(value{ii}, ...
                        toGPU = pv.toGPU, force=pv.force);
                    if self.verbose; fprintf('\t %s changed.\n', prop_name{ii}); end
                    self.clear_computed_properties(prop_name{ii});

                end

            catch e

                aa
            end

        end

    end


    methods (Access = protected, Static)

        function J = solve_jacobian_(varargin)
            % This is a protected wrapper for the Symbolic Math Toolbox's
            % jacobian function to prevent naming conflicts.
            J = jacobian(varargin{:}); % from Symbolic Math Toolbox
        end

        function func = expr2func_(expr, input_order, pv)

            arguments
                expr {mustBeA(expr, 'sym')}
                input_order (1,:) cell
                pv.returnSparse = false
            end

            func = matlabFunction(expr, Vars = input_order, Sparse=pv.returnSparse);
            
            hasX = has(expr, 'x');
            hasNull = has(expr,'null');
            hasVecInp = hasX | hasNull;
            if ~any(hasVecInp(:)), return; end

            % Now fix the case where (1) expr is a matrix, and (2)
            % at least one of the expressions in expr is x in which case we
            % need to handle the output shape when 'x' is not a scalar

            try
            if any(hasX(:))
                batch_var = 'x';
            elseif any(hasNull(:))
                batch_var = 'null';
            end
            
            func_str = func2str(func);
            if contains(func_str,'reshape')
                func_str = edit_reshape_expr_(func_str, batch_var);                
            end

            if contains(func_str,'sparse')
                func_str = edit_sparse_expr_(func_str, batch_var);
            end
            func = str2func(func_str);
            catch e
                aa
            end
            % hasX = has(expr, 'x');
            
            % func_str = func2str(func);
            % if contains(func_str,'reshape') && any(hasVecInp(:))
            % 
            %     % now find the reshape argument and change its size input
            %     pattern = 'reshape\s*\(\s*\[.*?\]\s*,\s*(\[.*?\]|[^,\)]+)';
            %     [~, tokenIndices] = regexp(func_str, pattern, 'tokens', 'tokenExtents');
            %     assert(isscalar(tokenIndices), "The solver function was" + ...
            %         " created erroneously with either zero or multiple `reshape` calls!");
            % 
            %     %insert size argument
            %     insert_idx = tokenIndices{1}(1);
            %     % check if it works with size(x) as well, currently x must
            %     % be a vector
            %     func_str = [func_str(1:insert_idx), 'numel(x),', ...
            %         func_str(insert_idx+1:end)];
            %     func = str2func(func_str);
            % 
            % end

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
            % if it is a matrix make sure that it outputs 3D array when x
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

% Following functions fix functions created by solvers to handle non-scalar
% entries (when x or null is a vector)
% matlabFunction outputs a function handle that assumes the inputs are all scalar
function func_str = edit_reshape_expr_(func_str, batch_var)
% EDIT_RESHAPE_EXPR_ Injects a batch dimension into reshape calls.
%
%   func_str = edit_reshape_expr_(func_str, 'x')
%
%   Transforms: reshape(vals, [m, n])
%   Into:       reshape(vals, numel(x), [m, n])
%
%   This ensures that when 'vals' is a vectorized batch (N x M*N),
%   the output becomes a 3D array (N x M x N) or similar, preserving
%   the batch dimension.

arguments
    func_str char
    batch_var char = 'x' % Default batch variable name
end

% Pattern to find the size argument of a reshape call.
% Matches: reshape(..., [size_args] OR size_var )
% We capture the start of the second argument (the size).
pattern = 'reshape\s*\(\s*(?:\[.*?\]|[^,]+)\s*,\s*(\[.*?\]|[^,\)]+)';

[~, tokenIndices] = regexp(func_str, pattern, 'tokens', 'tokenExtents');
if ~isempty(tokenIndices)
    % We only expect one reshape per function in this context,
    % but let's handle the first one found.
    idx = tokenIndices{1}(1);

    % Check if the size argument is just '1' (scalar case),
    % which might need special handling, but usually we just prepend.

    % Inject 'numel(batch_var),' before the existing size argument
    injection = sprintf('numel(%s), ', batch_var);

    func_str = [func_str(1:idx), injection, func_str(idx+1:end)];
end

end

function func_str = edit_sparse_expr_(func_str, batch_var)
    % EDIT_SPARSE_EXPR_ Converts sparse() calls to 3D array reshape() calls.
    %
    %   Transforms: sparse(rows, cols, vals, m, n)
    %   Into:       reshape(vals, numel(batch_var), m, n)
    %
    %   Output Size: (N x m x n)  <-- A stack of dense matrices
    
    arguments
        func_str char
        batch_var char = 'null' 
    end

    % Pattern to capture the 5 arguments of sparse
    % 1: Rows, 2: Cols, 3: Values, 4: m, 5: n
    pat = 'sparse\s*\(\s*(\[[\d\s,]+\])\s*,\s*(\[[\d\s,]+\])\s*,\s*(.+?)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)';
    
    % Find start/end of the match to ensure clean replacement
    [start_idx, end_idx] = regexp(func_str, pat, 'start', 'end', 'once');
    tokens = regexp(func_str, pat, 'tokens', 'once');
    
    if ~isempty(tokens) && ~isempty(start_idx)
        % We don't need rows_lit or cols_lit if we assume full dense output
        val_expr = tokens{3}; 
        m_lit    = tokens{4}; 
        n_lit    = tokens{5};
        
        % The New Code: reshape(values, N, m, n)
        % This creates an N x 2 x 2 array.
        new_code = sprintf('reshape(%s, numel(%s), %s*%s)', ...
                           val_expr, batch_var, m_lit, n_lit);

        new_code = sprintf('sparse(reshape(%s, numel(%s), %s*%s))', ...
                           val_expr, batch_var, m_lit, n_lit);
        
        % Replace the entire sparse(...) string with the new reshape(...) string
        func_str = [func_str(1:start_idx-1), ...
                    new_code, ...
                    func_str(end_idx+1:end)];
    end
end