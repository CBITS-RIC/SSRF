clear all;
close all;

subjects = [1 2 3];   %for few-samples and cross-trial (subject no.)
                    %each row is for one run, columnns are subjects in a multi-subject run
subj_lab = 1; %for cross-subject (labeled)
subj_unlab = 2:30;   %for cross-subject (unlabeled)
Nsamp = 1;  %for few-samples (no. samples per trial)
RepFac = 1; %for few-samples (1 = equal; 0 = no balance)
trial = 1;  %for cross-trial (trial no. for labeled)

ntrees = 5;
epochs = 20;
T0 = 1;
tau = 40;
alpha = 1;

X = load('../Data/UCIHARDataset/train/X_train.txt');
X = [X; load('../Data/UCIHARDataset/test/X_test.txt')];
Y = load('../Data/UCIHARDataset/train/y_train.txt');
Y = [Y; load('../Data/UCIHARDataset/test/y_test.txt')];
subjects_all = load('../Data/UCIHARDataset/train/subject_train.txt');
subjects_all = [subjects_all; load('../Data/UCIHARDataset/test/subject_test.txt')];

for i = 1:size(subjects,1),
    subj = subjects(i,:);
    disp(['******* Subject(s) ', subj]);
    acc_{i} = [];
    for k = 1:10,
        tic;
        fprintf('\n******* Iteration %d:\n', k);
        
%         [Xl Yl Xu Yu] = config_fewsamples(X, Y, subjects_all, subj, Nsamp, RepFac);
        %     [Xl Yl Xu Yu] = config_crosstrial(X, Y, subjects_all, subj, trial);
            [Xl Yl Xu Yu] = config_crosssubject(X, Y, subjects_all, subj_lab, subj_unlab);
        
        clear trees;
        [trees acc] = ssrf(Xl, Yl, Xu, Yu, ntrees, epochs, T0, tau, alpha);
        acc_{i} = [acc_{i}; acc];
        toc;
    end
end

figure;
for i=1:size(subjects,1),
%     subplot(size(subjects,1)/5, 5, i);
    bar([acc_{i}(:,1), acc_{i}(:,end)]);
    imp(:,i) = (acc_{i}(:,end) - acc_{i}(:,1))./acc_{i}(:,1);
    ylim([.5 1]);
    xlim([0 k+1]);
end

figure;
for i=1:size(subjects,1),
%     subplot(size(subjects,1)/5, 5, i);
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