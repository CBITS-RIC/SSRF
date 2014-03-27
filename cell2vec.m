function [vec labels] = cell2vec(y_cell)

% this function converts the strings in y_cell to integers,
% which can be changed back to y_cell using labels(vec)

labels = unique(y_cell);
vec = zeros(size(y_cell));
for i = 1:length(vec)
    if iscell(y_cell)
      [dummy, loc] = ismember(y_cell{i}, labels);
    else
      [dummy, loc] = ismember(y_cell(i), labels);        
    end
    vec(i) = loc;
end
