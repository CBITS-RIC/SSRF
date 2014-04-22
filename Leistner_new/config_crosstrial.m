function [Xl Yl Xu Yu] = config_crosstrial(X, Y, subjects, subj, trial)

subject_codes = unique(subjects);

%choosing the first training subject -- this is an example
subj = subject_codes(subj);
ind = find(subjects==subj);
X = X(ind, :);
Y = Y(ind);

%remove standing %%%%%%%%%%%%%%%%%%%%%%%
X(Y == 5,:) = [];
Y(Y == 5) = [];
Y(Y == 6) = 5;              %fix the class codes to fill the gap left

classes = unique(Y);         %the class codes
n_class = length(classes); %the # of classes

ind_sample = [];

for i=1:n_class,
    ind_class = find(Y==classes(i));
    
    %finding the middle sample in each trial
    trans_end = find(diff(ind_class)>1);
    trans_start = trans_end+1;
    trans_start = [ind_class(1), ind_class(trans_start)'];
    trans_end = [ind_class(trans_end)', ind_class(end)];

    ind_sample = [ind_sample, trans_start(trial):trans_end(trial)];

end

Xl = X(ind_sample,:);
Yl = Y(ind_sample);

inds = 1:size(X,1);
ind_nosample = inds(~ismember(inds, ind_sample));

Xu = X(ind_nosample, :);
Yu = Y(ind_nosample);

disp('Labeled/Unlabeled data configured:');
fprintf('%d labeled samples\n', length(ind_sample));   
fprintf('%d unlabeled samples\n', length(ind_nosample));