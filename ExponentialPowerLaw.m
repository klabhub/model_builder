classdef ExponentialPowerLaw < ModelBuilder
    % ExponentialPowerLaw defines a model of the form:
    %   y = intercept * x^-exponent * 10^(-x / knee)
    %
    % This class inherits from ModelBuilder to leverage its symbolic and
    % numerical computation engine, as well as its plotting capabilities.

    properties (Constant)
        % Symbolic constants for parameter names, ensuring consistency.
        intercept = sym('intercept');
        exponent = sym('exponent');
        knee = sym('knee');

    end

    properties
        % --- Abstract Properties Implementation from ModelBuilder ---
        model sym % The main symbolic function, defined in the constructor
        cacheModel = true

    end

    properties (Dependent)
        % --- Abstract Dependent Property Implementation from ModelBuilder ---
        parameters % Vector of symbolic parameters for this model

        includeKnee (1,1) logical % Flag to include the exponential knee term
        inLogScale (1,1) logical % Flag to compute on a log scale
        lower_bounds
        upper_bounds

    end

    properties (Access = protected)

        lower_bounds_ = []
        upper_bounds_ = []
        inLogScale_ = true
        includeKnee_ = false

    end

    methods
        function self = ExponentialPowerLaw(pv)
            % Constructor for the ExponentialPowerLaw class.
            % It defines the symbolic model and its configuration.
            arguments
                pv.includeKnee (1,1) logical = true
                pv.inLogScale (1,1) logical = true
                pv.cacheModel (1,1) logical = false
                pv.verbose (1,1) logical = true
            end
            % Assign configuration from name-value pairs
            self.verbose = pv.verbose;
            self.cacheModel = pv.cacheModel;
            self.inLogScale_ = pv.inLogScale;
            self.includeKnee_ = pv.includeKnee;            

            self.make_();

        end

        % --- Abstract Method Implementations from ModelBuilder ---
        function P_est = estimate(self, x_data, y_data)

            if ~isvector(y_data)

                y_data = median(y_data, 2);

            end
            kneeN = [];
            isBeforeKnee = false(size(x_data));
            isBeforeKnee(1:round(numel(x_data)/2)) = true;
            if self.includeKnee

                % check if the data is regularly-spaced
                if ~isscalar(unique(diff(x_data)))

                    % interpolate the breaks
                    [y_data, x_data] = self.interpolate_breaks_(y_data, x_data);

                end

                % The following function finds changes in the local slope. A knee would
                % exist at such conjunction


                % To estimate knee as a slope-change point, both x and y
                % has to be in log scale. However, this results in unequal
                % sampling in x axis. This invalidates the result of
                % findchangepts, since it expects a regularly spaced data.
                % we first need to upsample y_data to be regularly spaced
                % in log-log scale.

                log_x_data = linspace(log10(min(x_data)), log10(max(x_data)), numel(x_data))';
                y_interp = interp1(log10(x_data), y_data, log_x_data, 'linear');


                kneeN = 0;
                n_ch_pt = 1;                
                while kneeN < 20
                    % if found an earlier point, it is unlikely to be assc w knee parameter
                    % try again with more ch_pts
                    % knee_idx = findchangepts(y_interp, MaxNumChanges=n_ch_pt, Statistic= "linear");
                    % if ~isempty(knee_idx)
                    %     kneeN = 10^log_x_data(max(knee_idx));
                    % end
                    knee_idx = findchangepts(y_data, MaxNumChanges=n_ch_pt, Statistic= "linear");
                    if ~isempty(knee_idx)
                        kneeN = x_data(max(knee_idx));
                    end
                    n_ch_pt  = n_ch_pt + 1;
                end

                % Check if knee estimate is within predesignated bounds
                isBeforeKnee = x_data <= kneeN;
            end

            mdl_for_exp = fitlm(log10(x_data(isBeforeKnee)), y_data(isBeforeKnee));
            exponentN = -mdl_for_exp.Coefficients.Estimate(2);
            interceptN = mdl_for_exp.Coefficients.Estimate(1);

            P_est = [interceptN, exponentN, kneeN];

            lowerB = P_est<self.lower_bounds;
            upperB = P_est>self.upper_bounds;

            P_est(lowerB) = self.lower_bounds(lowerB);
            P_est(upperB) = self.upper_bounds(upperB);


        end

        % --- GET Methods ---
        function p = get.parameters(self)
            % Defines the ordered vector of symbolic parameters for this model.
            p = [self.intercept, self.exponent];
            if self.includeKnee
                p(end+1) = self.knee;
            end
        end
        
        % Calculate bounds based on data or call assigned bounds
        function b = get.lower_bounds(self)

             if isempty(self.X_) || isempty(self.Y_)
                 
                 b = [];

             elseif ~isempty(self.lower_bounds_)

                 b = self.lower_bounds_;

             else              

                 % intercept must be larger than median
                 y_data = self.Y_;
                 if self.inLogScale, y_data = 10.^y_data; end
                 int_lb = median(y_data, 'all');
                 exp_lb = .1;
                 b = [int_lb, exp_lb];
                 if self.includeKnee

                     knee_lb = max(min(self.X_), 5);
                     b = [b, knee_lb];

                 end

                 self.lower_bounds = b;
             
             end

         end

         function set.lower_bounds(self, value)
             
             if isempty(value), self.lower_bounds_ = []; return; end
             n_param = numel(value);
             if ~isempty(value) && self.n_param ~= n_param

                 error("Lower bound vector must be equal in length to " + ...
                     "the number of parameters, or left empty.")
                 
             end

             self.lower_bounds_ = value;

         end

         function b = get.upper_bounds(self)

             if isempty(self.X_) || isempty(self.Y_)
                 
                 b = [];

             elseif ~isempty(self.upper_bounds_)

                 b = self.upper_bounds_;

             else              
                                  
                 y_data = self.Y_;
                 if self.inLogScale, y_data = 10.^y_data; end
                 % intercept cannot be meaningful if too larger from the
                 % maximum amplitude
                 int_lb = max(y_data,[],'all')*1.25;
                 exp_lb = 10;
                 b = [int_lb, exp_lb];
                 if self.includeKnee

                     knee_lb = max(self.X_);
                     b = [b, knee_lb];

                 end
                 self.upper_bounds = b;
             
             end

         end
         
         function set.upper_bounds(self, value)
             
             if isempty(value), self.upper_bounds_ = []; return; end
             n_param = numel(value);
             if ~isempty(value) && self.n_param ~= n_param

                 error("Upper bound vector must be equal in length to " + ...
                     "the number of parameters, or left empty.")
                 
             end


             self.upper_bounds_ = value;

         end

         function set.includeKnee(self, val)

             arguments
                 self
                 val (1,1) logical                 
             end

             if val ~= self.includeKnee_

                 self.includeKnee_ = val;
                 self.make_();                

             end

         end
         function i = get.includeKnee(self)
             i = self.includeKnee_;
         end

         function set.inLogScale(self, val)

             arguments
                 self
                 val (1,1) logical                 
             end

             if val ~= self.inLogScale

                 self.inLogScale_ = val;
                 self.make_();                

             end

         end

         function i = get.inLogScale(self)
             i = self.inLogScale_;
         end
         
    end

    methods (Access = protected)

        function make_(self)
            % Generate a cache key based on the model configuration
            cacheKey = double(self.includeKnee_) + 2 * double(self.inLogScale_);

            % 1. Check if we have already compiled this model structure
            if self.cacheModel
                [cached_data, isCached] = ExponentialPowerLaw.manage_cache_(cacheKey);
            else
                isCached = false;
            end

            if isCached
                if self.verbose; fprintf('\t(Loading compiled model from cache...)\n'); end
                self.load_cache_(cached_data);
            else
                if self.verbose; fprintf('Constructing ExponentialPowerLaw model...\n'); end
                % Define the core power-law model
                base_model = self.intercept * (self.x^-self.exponent);
                % Optionally add the exponential knee term
                if self.includeKnee
                    self.model = base_model * exp(-self.x / self.knee);
                else
                    self.model = base_model;
                end
                % Optionally transform the entire model to log scale for fitting
                if self.inLogScale
                    self.model = log10(self.model);
                end
                self.solve_model();
                self.solve_jacobian();
                self.solve_hessian();

                % Cache the results if enabled
                if self.cacheModel
                    self.cache_(cacheKey);
                end
            end
            
            self.lower_bounds = [];
            self.upper_bounds = [];

        end

        function load_cache_(self, cached_data)
            % --- LOAD FROM CACHE ---
            % Restore Symbolic Properties
            % For EPL, parameter names are constants, but model is dynamic
            self.model = cached_data.sym_props.model;
            
            % Restore Compiled Function Handles
            self.model_func_    = cached_data.funcs.model;
            self.jacobian_func_ = cached_data.funcs.jacobian;
            self.hessian_func_  = cached_data.funcs.hessian;
            
            % Restore cached derivatives
            self.jacobian_ = cached_data.derivs.jacobian;
            self.hessian_  = cached_data.derivs.hessian;
        end
        
        function cache_(self, cacheKey)
            % --- SAVE TO CACHE ---
            data_to_cache = struct();
            
            % 1. Save Symbolic Definitions
            % No need to save intercept/exponent/knee as they are constants
            data_to_cache.sym_props.model = self.model;
            
            % 2. Save Compiled Functions
            data_to_cache.funcs.model    = self.model_func_;
            data_to_cache.funcs.jacobian = self.jacobian_func_;
            data_to_cache.funcs.hessian  = self.hessian_func_;
            
            % 3. Save Derivatives
            data_to_cache.derivs.jacobian = self.jacobian_;
            data_to_cache.derivs.hessian  = self.hessian_;
            
            % Store in static memory
            ExponentialPowerLaw.manage_cache_(cacheKey, data_to_cache);
        end
    end

    % --- Caching Methods ---
    methods (Static)
        function clear_cache()
            % Utility to wipe memory if needed
            clear ExponentialPowerLaw.manage_cache_;
        end
    end
    methods (Static, Access = protected)
        function [data, is_cached] = manage_cache_(key, new_data)
            % This variable persists in memory between function calls
            persistent epl_cache_
            % Initialize cache if it doesn't exist
            if isempty(epl_cache_)
                epl_cache_ = containers.Map('KeyType', 'double', 'ValueType', 'any');
            end
            % If new data is provided, save it (Setter Mode)
            if nargin > 1
                epl_cache_(key) = new_data;
            end
            % Check if data exists (Getter Mode)
            if epl_cache_.isKey(key)
                data = epl_cache_(key);
                is_cached = true;
            else
                data = [];
                is_cached = false;
            end
        end
    end
end
