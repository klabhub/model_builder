function [matched_params, nonmatching_params] = synch_peak_parameters_(target, source, tol)
% SYNCH_PEAKS_ Matches source peaks to target peaks based on center frequency.
%
%   [matching_target, nonmatching_target, nonmatching_source] = ...
%       synch_peaks_(target, source, tol)
%
% Arguments:
%   target (1,:) : Flattened peak parameters [amp1, amp2..., freq1, freq2..., sd1, sd2...]
%   source (1,:) : Flattened peak parameters [amp1, amp2..., freq1, freq2..., sd1, sd2...]
%   tol (1,1)    : Frequency tolerance for matching (default = 2)
%
% Returns:
%   matching_target    : Target parameters that found a match.
%   nonmatching_target : Target parameters that did not find a match.
%   nonmatching_source : Source parameters that did not find a match.

arguments
    target (1,:)
    source (1,:)
    tol (1,1) = 2
end

% 1. Reshape inputs so each row represents a peak: [Amp, Freq, SD]
% With input format [Amps..., Freqs..., SDs...], reshape(vec, [], 3)
% fills columns first, correctly assigning Amps to col 1, Freqs to col 2, etc.
target_mat = reshape(target, [], 3);
source_mat = reshape(source, [], 3);

n_target = size(target_mat, 1);
n_source = size(source_mat, 1);

% 2. Create Distance Matrix comparing Center Frequencies (2nd column)
% Rows = Target indices, Cols = Source indices
if n_target > 0 && n_source > 0
    t_freqs = target_mat(:, 2);
    s_freqs = source_mat(:, 2)';
    dist_matrix = abs(t_freqs - s_freqs);
else
    matched_params = [];
    nonmatching_params = [target, source];
    return
end

% Keep track of which indices have been matched
target_matched_mask = false(n_target, 1);
source_matched_mask = false(n_source, 1);

% 3. Greedy Matching Algorithm
% Iterate until no matches < tol are found or one list is exhausted
while true
    if isempty(dist_matrix)
        break;
    end

    % Find the minimum distance in the entire matrix
    [min_val, linear_idx] = min(dist_matrix(:));

    % If the smallest distance exceeds tolerance, we are done
    if min_val > tol
        break;
    end

    % Convert linear index back to (row, col) -> (target_idx, source_idx)
    [t_idx, s_idx] = ind2sub(size(dist_matrix), linear_idx);

    % Record the match
    target_matched_mask(t_idx) = true;
    source_matched_mask(s_idx) = true;

    % Set the distance for this row (target) and column (source) to infinity
    % so they cannot be matched again.
    dist_matrix(t_idx, :) = inf;
    dist_matrix(:, s_idx) = inf;
end

% 4. Separate and Reshape Outputs

% Get matching rows
m_target_rows = target_mat(target_matched_mask, :);

% Convert back to [Amps..., Freqs..., SDs...] format
% The linear index (:) of an Mx3 matrix stacks columns: Col1, then Col2, then Col3.
% Transpose to return a 1xN row vector.
matched_params = m_target_rows(:)';

% Get non-matching rows and reshape
nm_target_rows = target_mat(~target_matched_mask, :);
% nonmatching_target = nm_target_rows(:)';

nm_source_rows = source_mat(~source_matched_mask, :);

% append nonmatching
nonmatching_params = nm_source_rows;%[nm_source_rows; nm_target_rows];
nonmatching_params = nonmatching_params(:)';
% nonmatching_source = nm_source_rows(:)';

end