clear all;
close all;

subj = 1:10;
plot_transitions = true;

X = load('../Data/UCIHARDataset/train/X_train.txt');
Y = load('../Data/UCIHARDataset/train/y_train.txt');

act_labels = {'walk','upstairs','downstairs','sit','stand','lay'};
% red, magenta, yellow, blue, green, cyan
colors = {[1 0 0], [1 0 1], [1 1 0], [0 0 1], [0 1 0], [0 1 1]};

subjects = load('../Data/UCIHARDataset/train/subject_train.txt');
subject_codes = unique(subjects);
subj = subject_codes(subj);
ind = [];
for i=1:length(subj),
    ind = [ind; find(subjects==subj(i))];
end
X = X(ind, :);
Y = Y(ind);

X_red = tsne(X);
scatter(X_red(:,1), X_red(:,2), 4, reshape(cell2mat(colors(Y)),3,length(cell2mat(colors(Y)))/3)');