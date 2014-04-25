function [Xl Yl Xu Yu] = config_fewsamples(X, Y, subjects, subj, Nsamp, RepFac)

subject_codes = unique(subjects);

%choosing the first training subject -- this is an example
subj = subject_codes(subj);
ind = [];
for s=1:length(subj),
    ind = [ind; find(subjects==subj(s))];
end
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
    
    for n = 1:Nsamp
        for j = 1:length(trans_start),
            ind_sample = [ind_sample randi([trans_start(j), trans_end(j)],1)];
        end
    end

end

Xl = X(ind_sample,:);
Yl = Y(ind_sample);

inds = 1:size(X,1);
ind_nosample = inds(~ismember(inds, ind_sample));

Xu = X(ind_nosample, :);
Yu = Y(ind_nosample);

% repeating training data to balance the total number of labeled vs
% unlabeled
Xl = repmat(Xl, 1+round(RepFac*length(Yu)/length(Yl)), 1);
Yl = repmat(Yl, 1+round(RepFac*length(Yu)/length(Yl)), 1);

disp('Labeled/Unlabeled data configured:');
fprintf('%d labeled samples\n', length(ind_sample));   
fprintf('%d unlabeled samples\n', length(ind_nosample));
fprintf('%d labeled samples after balancing\n', length(Yl));