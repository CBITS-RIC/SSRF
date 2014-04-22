clear all;
close all;

subj = 1;
Nsamp = 2;
RepFac = 1;

Ntrees = 1000;
epochs = 10;
T0 = 1;
tau = 40;
alpha = 1;

X = load('../Data/UCIHARDataset/train/X_train.txt');
Y = load('../Data/UCIHARDataset/train/y_train.txt');

acc_ = [];
for k = 1:3,
    k
    [Xl Yl Xu Yu] = config_fewsamples(X, Y, subj, Nsamp, RepFac);
    clear trees;
    [trees acc] = ssrf(Xl, Yl, Xu, Yu, Ntrees, epochs, T0, tau, alpha);
    acc_ = [acc_; acc];
end

bar([acc_(:,1) , acc_(:,end)]);