clear all;
close all;

subj = 3;   %for few-samples and cross-trial (subject no.)
subj_lab = 1; %for cross-subject (labeled)
subj_unlab = 2:10;   %for cross-subject (unlabeled)
Nsamp = 2;  %for few-samples (no. samples)
RepFac = 1; %for few-samples (1 = equal; 0 = no balance)
trial = 1;  %for cross-trial (trial no. for labeled)

ntrees = 10;
epochs = 50;
T0 = 1;
tau = 40;
alpha = 1;

X = load('../Data/UCIHARDataset/train/X_train.txt');
Y = load('../Data/UCIHARDataset/train/y_train.txt');
subjects = load('../Data/UCIHARDataset/train/subject_train.txt');

acc_ = [];
for k = 1:10,
    fprintf('\n******* Iteration %d:\n', k);
    
%     [Xl Yl Xu Yu] = config_fewsamples(X, Y, subjects, subj, Nsamp, RepFac);
%     [Xl Yl Xu Yu] = config_crosstrial(X, Y, subjects, subj, trial);
    [Xl Yl Xu Yu] = config_crosssubject(X, Y, subjects, subj_lab, subj_unlab);
    
    clear trees;
    [trees acc] = ssrf(Xl, Yl, Xu, Yu, ntrees, epochs, T0, tau, alpha);
    acc_ = [acc_; acc];
end

figure;
acc_init = acc_(:,1);
acc_good = acc_(:,end); acc_good(acc_good<acc_init) = 0;
acc_bad = acc_(:,end); acc_bad(acc_bad>=acc_init) = 0;
bar([acc_init, acc_good]);
colormap([0 0 1; 0 1 0]);
figure;
bar([acc_init, acc_bad]);
colormap([0 0 1; 1 0 0]);
