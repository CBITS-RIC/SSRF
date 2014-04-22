function [Xl Yl Xu Yu] = config_crosssubject(X, Y, subjects, subj_lab, subj_unlab)

subject_codes = unique(subjects);

%remove standing %%%%%%%%%%%%%%%%%%%%%%%
X(Y == 5,:) = [];
Y(Y == 5) = [];
Y(Y == 6) = 5;              %fix the class codes to fill the gap left

%choosing the first training subject -- this is an example
subj = subject_codes(subj_lab);
ind = [];
for i=1:length(subj),
    ind = [ind; find(subjects==subj(i))];
end
Xl = X(ind, :);
Yl = Y(ind);

subj = subject_codes(subj_unlab);
ind = [];
for i=1:length(subj),
    ind = [ind; find(subjects==subj(i))];
end
Xu = X(ind, :);
Yu = Y(ind);

disp('Cross-subject data created.');
fprintf('%d labeled samples\n', length(Yl));   
fprintf('%d unlabeled samples\n', length(Yu));