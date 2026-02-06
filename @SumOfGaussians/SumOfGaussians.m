classdef  SumOfGaussians < ModelBuilder

    properties

        baseline (1,1) sym = sym('b')
        amplitude (:,1) sym = sym('a')
        center (:,1) sym = sym('mu')
        sd (:,1) sym = sym('s')
        n_peaks (1,1) {mustBeInteger}

        min_peak_width = .1
        min_peak_distance = 1
        min_peak_frequency = NaN
        max_peak_frequency = NaN
        max_peak_frequency_offset = NaN % if set changes lower and upper bounds for center_frequency estimates


        includeBaseline = true
        
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
        lower_bounds
        upper_bounds
               
     end

     properties (Access = protected)

         lower_bounds_ = []
         upper_bounds_ = []

     end

     methods (Static)

         [matching_target, nonmatching_target, nonmatching_source] = synch_peak_parameters_(target, source, tol)

     end
     methods

         function self = SumOfGaussians(pv)

            % Constructor for the ExponentialPowerLaw class.
            % It defines the symbolic model and its configuration.
            arguments
                
                pv.n_peaks (1,1) {mustBeInteger, mustBePositive} = 1
                pv.min_peak_width = []
                pv.min_peak_distance = []

                pv.min_peak_frequency = []
                pv.max_peak_frequency = []
                pv.max_peak_frequency_offset = []
                
                pv.verbose (1,1) logical = true
                pv.includeBaseline (1,1) logical = true;

            end
            self.verbose = pv.verbose;
            
            if ~isempty(pv.min_peak_distance)

                self.min_peak_distance = pv.min_peak_distance;

            end

            if ~isempty(pv.min_peak_width)

                self.min_peak_width = pv.min_peak_width;

            end

            if ~isnan(pv.min_peak_frequency)

                self.min_peak_frequency = pv.min_peak_frequency;

            end

            if ~isnan(pv.max_peak_frequency)

                self.max_peak_frequency = pv.max_peak_frequency;

            end

            if ~isnan(pv.max_peak_frequency_offset)
                self.max_peak_frequency_offset = pv.max_peak_frequency_offset;
            end

            self.includeBaseline = pv.includeBaseline;
            
            % triggers solving symbolic functions
            self.n_peaks = pv.n_peaks;

            
            % --- Model Definition ---
            if self.verbose; fprintf('Constructing Gaussian model...\n'); end
         end
      
         function set.n_peaks(self, value)

             try

             % if changes, the model changes
             self.n_peaks = value;
             self.make_();

             catch e
                 a
             end

         end     
         

         function p_est = estimate(self, x_data, y_data,pv)

             arguments

                 self
                 x_data (:, 1) double = self.X
                 y_data (:, :) double = self.Y
                 
                 pv.min_peak_width (1, 1) double = self.min_peak_width
                 pv.min_peak_distance(1,1) double = self.min_peak_distance

                 pv.min_peak_frequency = self.min_peak_frequency
                 pv.max_peak_frequency = self.max_peak_frequency                

             end

             % If y_data contains multiple observations, take the mean
             if size(y_data,2) > 1
                 y_data = median(y_data, 2);
             end

             % Constrain peak search to pre-assigned bounds
             xInPeakSearch = true(size(x_data));
             if ~isnan(pv.min_peak_frequency)
                 xInPeakSearch = xInPeakSearch & (x_data >= pv.min_peak_frequency);
             end

             if ~isnan(pv.max_peak_frequency)
                 xInPeakSearch = xInPeakSearch & (x_data <= pv.max_peak_frequency);
             end

             x_data = x_data(xInPeakSearch);
             y_data = y_data(xInPeakSearch);

             % Linear interp if freqs contain breaks
             if ~isscalar(unique(diff(x_data)))

                 [y_interp, x_interp] = self.interpolate_breaks_(y_data, x_data);

             end
                         

             % estimate min prominence threshold
             % flatten by detrending
             dx = mode(diff(x_interp));
             y_flat = detrend(y_interp);
             % get Q3 and median (in case median is diff than zero)
             % [~, q] = iqr(y_flat); % does not work on R2023a
            
             med = median(y_flat);
             q2 = median(y_flat(y_flat > med));
             p_thr = q2-med;

            
             [amp, cf, bw, p] = findpeaks(...
                 y_interp, ...
                 x_interp, ...
                 MinPeakDistance = pv.min_peak_distance, ...
                 MinPeakProminence= p_thr,...
                 MinPeakWidth = pv.min_peak_width,...
                 NPeaks=self.n_peaks, SortStr='descend');

             cf_idx = arrayfun(@(f) do.argmin(abs(x_interp - f)), cf);
             bw_bins = bw./dx;
             bw_bins(bw_bins < 2) = 2; % at least 2 bins necessary
             
             % refine peak estimates for gaussian fitting
             % The peaks must be centered in Gaussians but findpeaks does not force this
             % constraint.
             % Instead, we will find the change points around the peak locations to
             % determine peak onset and offset locs, then, we will re-estimate:
             % 1. center location as the mean of the onset and offset,
             % 2. prominence as the mean difference between the peak and the bases
             % 3. bandwidth as the half max width from the new peak location

             % Find the change points

             ch_pts_idx = findchangepts(y_interp, ...
                 MaxNumChanges = 6*self.n_peaks, ... 6*(onset, peak, offset) x max_n_peaks, higher no makes sure the peaks are detected
                 Statistic='linear'... slope changes
                 );
             ch_pts_freqs = x_interp(ch_pts_idx);
             ch_pts_amps = y_interp(ch_pts_idx);

             min_abs_slope_for_flat = .1;

             % Loop each peak center
             for iPk = 1:numel(amp)

                 cfN = cf(iPk);

                 nearest_ch_pt_idx = do.argmin(abs(ch_pts_freqs-cfN));

                 % onset slopes

                 pre_peak_coords = [ch_pts_freqs(1:nearest_ch_pt_idx-1), ch_pts_amps(1:nearest_ch_pt_idx-1)];
                 if size(pre_peak_coords,1) == 1
                     n_pts_from_onset_to_peak = 1;
                 else
                     pre_peak_slopes = flip(diff(pre_peak_coords), 1);
                     pre_peak_slopes = pre_peak_slopes(:,2) ./ pre_peak_slopes(:,1);
                     n_pts_from_onset_to_peak = find(pre_peak_slopes < min_abs_slope_for_flat, 1, 'first');

                 end
                 % offset slopes
                 post_peak_coords = [ch_pts_freqs(nearest_ch_pt_idx+1:end), ch_pts_amps(nearest_ch_pt_idx+1:end)];
                 if size(post_peak_coords,1) == 1
                     n_pts_from_offset_to_peak = 1;
                 else

                     post_peak_slopes = diff(post_peak_coords);
                     post_peak_slopes = post_peak_slopes(:,2) ./ post_peak_slopes(:,1);
                     n_pts_from_offset_to_peak = find(post_peak_slopes > -min_abs_slope_for_flat, 1, 'first');
                 end

                 iOnsetN = ch_pts_idx(nearest_ch_pt_idx - n_pts_from_onset_to_peak);
                 iOffsetN = ch_pts_idx(nearest_ch_pt_idx+n_pts_from_offset_to_peak);

                 % use the mean of new and old estimates
                 cf_new = (cfN + mean(x_interp([iOnsetN, iOffsetN])))/2;
                 cf_idx_new = do.argmin(abs(x_interp-cf_new));

                 left_half_amp = mean(y_interp([iOnsetN, cf_idx_new]));
                 right_half_amp = mean(y_interp([cf_idx_new, iOffsetN]));

                 local_amps = nan(size(x_interp));
                 local_amps(iOnsetN:cf_idx_new) = y_interp(iOnsetN:cf_idx_new);
                 left_half_max_idx = do.argmin(abs(local_amps - left_half_amp));

                 local_amps = nan(size(x_interp));
                 local_amps(cf_idx_new:iOffsetN) = y_interp(cf_idx_new:iOffsetN);
                 right_half_max_idx = do.argmin(abs(local_amps - right_half_amp));

                 bw_new = diff(x_interp([left_half_max_idx, right_half_max_idx]));

                 p_old = y_interp(cf_idx(iPk)) - mean(y_interp([iOnsetN, iOffsetN]));
                 p_new = (p_old + y_interp(cf_idx_new) - mean(y_interp([iOnsetN, iOffsetN])))/2;
                 % amp_new = spectrum(cf_idx_new);
                 % p_new = amp_new - mean(spectrum([iOnsetN, iOffsetN]));

                 cf(iPk) = cf_new;
                 amp(iPk) = p_new;
                 p(iPk) = (cf_new-cfN)/2;
                 % p(iPk) = p_new;
                 bw(iPk) = bw_new;

             end

             init_b = med; % baseline

             std = bw / (2*sqrt(2*log(2))); % bw to sigma
             init_params = [amp, cf, std];


             % sorting in descending order of amplitude
             init_params = sortrows(init_params, 1, "descend");
             p_est = [init_params(:); init_b]';
             
         end

         % Get Method
         function p = get.parameters(self)

             p = [self.amplitude, self.center, self.sd];
             p = [p(:); self.baseline];
             p = reshape(p, [1, numel(p)]);

         end

         function b = get.lower_bounds(self)

             if isempty(self.X_) || isempty(self.Y_)
                 b = [];

             elseif ~isempty(self.lower_bounds_)

                 b = self.lower_bounds_;

             else              

                 % since the peak amplitudes show 1/f-like decrease, it is
                 % better to be lenient in lower bounds
                 amp_lb = max(0, median(self.Y_, 'all'));

                 if isnan(self.min_peak_frequency)
                    cf_lb = min(self.X_) + self.min_peak_width; % must have a reasonable number of data points at least
                 else
                     cf_lb = self.min_peak_frequency;
                 end
                 sd_lb = 0;%self.min_peak_width / sqrt(2*log(2)); % transform from fwhm to sd
                 
                 b_lb = -.1;

                 b = [amp_lb, cf_lb, sd_lb, b_lb];
                 self.lower_bounds = b;
             
             end

         end

         function set.lower_bounds(self, value)
             
             if isempty(value), self.lower_bounds_ = []; return; end
             n_param =numel(value);
             if ~isempty(value) && self.n_param ~= n_param

                 value = [repelem(value(1:end-1), self.n_peaks), value(end)];

             end

             if ~isnan(self.max_peak_frequency_offset) && n_param && self.n_param
                 n_peak = floor(n_param/3);
                 cf_idx = n_peak+1:2*n_peak;
                 min_cf = value(cf_idx(1));
                 value(cf_idx) = self.P(cf_idx) - self.max_peak_frequency_offset;
                 value(cf_idx(value(cf_idx) < min_cf)) = min_cf;

             end

             self.lower_bounds_ = value;

         end

         function b = get.upper_bounds(self)

             if isempty(self.X_) || isempty(self.Y_)
                 
                 b = [];

             elseif ~isempty(self.upper_bounds_)

                 b = self.upper_bounds_;

             else              
                 
                 amp_ub = max(self.Y_(:))*1.5;

                 if isnan(self.max_peak_frequency)

                    cf_ub = max(self.X_) - mode(diff(self.X_))*2;

                 else

                     cf_ub = self.max_peak_frequency;

                 end
                 
                 sd_ub = diff(do.range(self.X_))/3 / (2*sqrt(2*log(2)));
                 
                 b_ub = .1;

                 b = [amp_ub, cf_ub, sd_ub, b_ub];
                 self.upper_bounds = b;
              
             end

         end
         
         function set.upper_bounds(self, value)

             if isempty(value), self.upper_bounds_ = []; return; end
             n_param = numel(value);
             if ~isempty(value) && self.n_param ~= n_param

                 value = [repelem(value(1:end-1), self.n_peaks), value(end)];

             end

             if ~isnan(self.max_peak_frequency_offset) && n_param && self.n_param
                 n_peak = floor(n_param/3);
                 cf_idx = n_peak+1:2*n_peak;
                 max_cf = value(cf_idx(1));
                 value(cf_idx) = self.P(cf_idx) + self.max_peak_frequency_offset;
                 value(cf_idx(value(cf_idx) > max_cf)) = max_cf;

             end

             self.upper_bounds_ = value;

         end
         
         % --- Overwrite Methods ---
         function y_sim = simulate(self, peak_params, baseline, sigma, varargin)

             if isvector(peak_params)
                 
                 peak_params = [peak_params; baseline];
             
             else
                 
                 peak_params = [peak_params(:); baseline]';

             end

             y_sim = simulate@ModelBuilder(self, peak_params, sigma, varargin{:});             


         end

         function YHat = predict(self, P, pv)%peak_params, varargin)

             arguments
                 self
                 P (1,:) double = self.P
                 pv.peaks = []
                 pv.baseline double = []
                 pv.X = self.X
                 % peak_params = self.P
             end

             if ~isempty(pv.peaks)
                 assert(~mod(numel(pv.peaks),3), "peaks must be n_peaks by 3!")
                 peak_params = pv.peaks(:);
             else
                 peak_params = P;
             end

             if ~isempty(pv.baseline)
                 assert(isscalar(pv.baseline), "baseline must be a scalar!")
                 peak_params(end+1) = pv.baseline;
             end

             if numel(peak_params) ~= self.n_param
                 self = self.copy();
                 self.n_peaks = floor(numel(peak_params)/3);
                 warning("Current model has different number of parameters" + ...
                     " then it was asked to predict.\n The prediction is" + ...
                     "based on a temporary model.")
             end
             YHat = predict@ModelBuilder(self, peak_params, X=pv.X);

             % arguments (Repeating)
             %     varargin
             % end

             % if isempty(peak_params), peak_params = self.P; end
             % if isvector(peak_params) 
             % 
             %     if ~mod(numel(peak_params), 3)
             % 
             %         peak_params = [peak_params; varargin{1}]'; %append baseline
             %         varargin(1) = [];
             % 
             %     end
             % 
             % else
             % 
             %     peak_params = [peak_params(:); varargin{1}]';
             %     varargin(1) = [];
             % 
             % end
             % 
             % varargin = [peak_params, varargin];
             % %update the model
             % if numel(peak_params)
             %     self.n_peaks = (numel(peak_params)-1)/3;
             % end
             % 
             % 
             % YHat = predict@ModelBuilder(self, varargin{:});

         end

         % Create inequality constraints for peak center frequency order
         function [A, b] = return_linineq_for_peak_centers(self, p0, min_peak_dist)

             arguments

                 self
                 p0 (1,:)

                 min_peak_dist = self.min_peak_distance
                 
             end

             n = self.n_peaks;
             if n > 1
                 A = zeros(n - 1, numel(p0));
                 % cf_i - cf_{i+1} <= -min_peak_dist where cf_i < cf_{i+1} 
                 b = zeros(n - 1, 1) - min_peak_dist;
    
                 % create cf matrix as if peaks are ordered ascending in
                 % center frequency
    
                 % the following will create rows like 
                 % [[..,0,] 1, -1, [0,...]]
                 A_cf = eye(n - 1, n);
                 if n > 2
                    A_cf(:, 2:end) = A_cf(:, 2:end) - eye(n-1, n-1);
                 else
                     A_cf(end) = -1;
                 end
                
                 % subset center frequency parameters
                 ii_cf = n + (1:n);
    
                 [~, sort_idx] = sort(p0(ii_cf), 'ascend');
                 A_cf(:, sort_idx) = A_cf;
    
                 A(:,ii_cf) = A_cf;
             else
                 A = [];
                 b = [];
             end

         end

         
     end

     methods (Access =protected)

         % Makes the model
         function make_(self)

            self.amplitude = sym('a', [self.n_peaks, 1]);
            self.center = sym('mu', [self.n_peaks, 1]);
            self.sd = sym('s', [self.n_peaks, 1]);
            self.model = sum(self.amplitude .* exp(-(self.x - self.center).^2 ./ (2 * self.sd.^2))) + self.baseline;
            self.solve_jacobian();
            self.solve_hessian();
            self.lower_bounds = [];
            self.upper_bounds = [];

         end
         
     end


end