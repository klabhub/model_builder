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
        x = sym('x'); % Symbolic independent variable
        y = sym('y'); % Symbolic dependent variable
        y_hat = sym('y_hat'); % Symbolic representation of the model's prediction

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
                pv.verbose (1,1) logical = true
            end
            % Assign configuration from name-value pairs
            self.verbose = pv.verbose;
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

        function self = make_(self)

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

            self.solve_jacobian();
            self.solve_hessian();
            self.lower_bounds = [];
            self.upper_bounds = [];

        end
    end

end
