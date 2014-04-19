clear all;
close all;

subj = 6;
plot_transitions = true;

X = load('../Data/UCIHARDataset/train/X_train.txt');
Y = load('../Data/UCIHARDataset/train/y_train.txt');
act_labels = {'walk','upstairs','downstairs','sit','stand','lay'};

subjects = load('../Data/UCIHARDataset/train/subject_train.txt');
subject_codes = unique(subjects);
subj = subject_codes(subj);
ind = find(subjects==subj);
X = X(ind, :);
Y = Y(ind);

figure(1);
subplot(3,1,[1 2]);
imagesc(X');
colormap gray;

if plot_transitions,
    trans = find(diff(Y)~=0);
    if ~isempty(trans),
        hold on;
        plot((trans*ones(1,2))', (ones(length(trans),1)*[0 size(X,2)])', ...
            'g', 'linewidth', 1);
    end
end

subplot(3,1,3);
plot(Y, '.');
set(gca, 'ytick', 1:6, 'yticklabel', act_labels);
axis('tight');