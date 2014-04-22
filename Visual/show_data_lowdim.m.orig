clear all;
close all;

<<<<<<< HEAD
subj = 1:4;
=======
subj = 5;
>>>>>>> 0eb7682f533e9c0d0772dcb772190d67a22a9455
plot_transitions = true;

X = load('../Data/UCIHARDataset/train/X_train.txt');
Y = load('../Data/UCIHARDataset/train/y_train.txt');

act_labels = {'walk','upstairs','downstairs','sit','stand','lay'};
% red, magenta, black, blue, green, cyan
colors = {[1 0 0], [1 0 1], [0 0 0], [0 0 1], [0 1 0], [0 1 1]};

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
scatter(X_red(:,1), X_red(:,2), 20, reshape(cell2mat(colors(Y)),3,length(cell2mat(colors(Y)))/3)');
