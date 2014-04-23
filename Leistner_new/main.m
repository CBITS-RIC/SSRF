clear all;
close all;

subj = 1:30;   %for few-samples and cross-trial (subject no.)
subj_lab = 2; %for cross-subject (labeled)
subj_unlab = 1;   %for cross-subject (unlabeled)
Nsamp = 1;  %for few-samples (no. samples per trial)
RepFac = 1; %for few-samples (1 = equal; 0 = no balance)
trial = 1;  %for cross-trial (trial no. for labeled)

ntrees = 500;
epochs = 20;
T0 = 1;
tau = 40;
alpha = 1;

X = load('../Data/UCIHARDataset/train/X_train.txt');
X = [X; load('../Data/UCIHARDataset/test/X_test.txt')];
Y = load('../Data/UCIHARDataset/train/y_train.txt');
Y = [Y; load('../Data/UCIHARDataset/test/y_test.txt')];
subjects = load('../Data/UCIHARDataset/train/subject_train.txt');
subjects = [subjects; load('../Data/UCIHARDataset/test/subject_test.txt')];

for subj_ind = subj,
    fprintf('\n******* Subject %d:\n', subj_ind);
    acc_{subj_ind} = [];
    for k = 1:10,
        tic;
        fprintf('\n******* Iteration %d:\n', k);
        
        [Xl Yl Xu Yu] = config_fewsamples(X, Y, subjects, subj_ind, Nsamp, RepFac);
        %     [Xl Yl Xu Yu] = config_crosstrial(X, Y, subjects, subj_ind, trial);
        %     [Xl Yl Xu Yu] = config_crosssubject(X, Y, subjects, subj_lab, subj_unlab);
        
        clear trees;
        [trees acc] = ssrf(Xl, Yl, Xu, Yu, ntrees, epochs, T0, tau, alpha);
        acc_{subj_ind} = [acc_{subj_ind}; acc];
        toc;
    end
end

figure;
for i=1:length(subj),
    subplot(length(subj)/5, 5, i);
    bar([acc_{i}(:,1), acc_{i}(:,end)]);
    imp(:,i) = (acc_{i}(:,end) - acc_{i}(:,1))./acc_{i}(:,1);
    ylim([.5 1]);
    xlim([0 k+1]);
end

figure;
for i=1:length(subj),
    subplot(length(subj)/5, 5, i);
    imp_good = imp(:,i);
    imp_bad = imp(:,i);
    imp_good(imp_good<0) = 0;
    imp_bad(imp_bad>=0) = 0;
    bar(imp_good, 'g');
    hold on;
    bar(imp_bad, 'r');
    xlim([0 k+1]);
end

figure;
boxplot(imp);

figure;
boxplot(imp(:));